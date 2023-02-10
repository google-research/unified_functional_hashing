# Copyright 2022 The functional_equivalence Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluator for NASBench.
"""

from typing import Dict, Optional, Tuple, Union

from unified_functional_hashing.nasbench.python import constants
from unified_functional_hashing.nasbench.python import evaluation
from unified_functional_hashing.nasbench.python import fec as fec_lib
from unified_functional_hashing.nasbench.python import noise_generator as noise_generator_lib
from nasbench import api
import numpy as np


class Evaluator():
  """Evaluator for NASBench ModelSpecs.
  """

  def __init__(self,
               nasbench: api.NASBench,
               fixed_run_idx: int = -1,
               mantissa_bits: int = 24,
               hashing_time: float = 10.0,
               noise_type: str = "",
               noise_stddev: float = 0.0,
               use_fec: bool = False,
               max_num_evals: int = 1,
               fec_remove_probability: float = 0.0,
               save_fec_history: bool = False,
               rng_seed: Optional[int] = None,
               test: bool = False):
    """Initializes Evaluator.

    Args:
      nasbench: NASBench instance.
      fixed_run_idx: The run index to use for getting computed_stats. If -1,
        then will be random and unfixed.
      mantissa_bits: number of bits to use in the mantissa.
      hashing_time: Number of seconds it takes to generate hash.
      noise_type: type of noise to add to the evaluated validation accuracy.
      noise_stddev: noise standard deviation for evaluated validation accuracy.
      use_fec: Whether to use FEC or not.
      max_num_evals: The maximum number of evaluations to perform for a hash.
      fec_remove_probability: After a cache hit, the probability a key will be
        removed from the cache.
      save_fec_history: Whether to save FEC's history. If True, the FEC's number
        of cache misses, cache partial hits, cache full hits, cache size, and
        the Hasher's 4 accuracy model hash will be saved.
      rng_seed: Seed to initialize rng.
      test: Whether called from a test or not.
    """
    self.nasbench = nasbench
    self.rng = np.random.default_rng(rng_seed)
    self.noise_generator = noise_generator_lib.NoiseGenerator(
        noise_type,
        noise_stddev,
        rng_seed=self.rng.integers(
            low=1, high=constants.MAX_RNG_SEED, size=1)[0])
    self.fec = None
    self.test = test
    self.fixed_run_idx = fixed_run_idx
    self.run_idx_set = {0, 1, 2}
    if use_fec:
      self.fec = fec_lib.FEC(
          nasbench=nasbench,
          noise_generator=self.noise_generator,
          mantissa_bits=mantissa_bits,
          hashing_time=hashing_time,
          max_num_evals=max_num_evals,
          remove_probability=fec_remove_probability,
          save_history=save_fec_history,
          rng_seed=self.rng.integers(
              low=1, high=constants.MAX_RNG_SEED, size=1)[0])

  def has_fec(self) -> bool:
    """Returns whether Evaluator has an FEC or not.

    Returns:
      Bool whether Evaluator has an FEC or not.
    """
    return self.fec is not None

  def get_fec_statistics(self) -> Dict[str, Union[int, str]]:
    """Gets FEC statistics.

    Returns:
      Dict of the number of cache misses, partial hits, and full hits, the
        current size of the cache, and possibly the graph and model hashes.
    """
    assert self.fec is not None, "Evaluator has no FEC!"
    return self.fec.get_cache_statistics()

  def evaluate(
      self, model_spec: api.ModelSpec) -> Tuple[int, float, float, float]:
    """Evaluates model spec to get eval outputs.

    Args:
      model_spec: ModelSpec matrix.

    Returns:
      4-tuple of int run_idx and floats for valid and test accuracies and
        training time.
    """
    if self.fec is None:
      if self.fixed_run_idx not in self.run_idx_set:
        run_idx = self.rng.integers(low=0, high=3, size=1)[0]
      else:
        run_idx = self.fixed_run_idx
      eval_outputs = evaluation.evaluate(
          nasbench=self.nasbench,
          noise_generator=self.noise_generator,
          model_spec=model_spec,
          run_idx=run_idx,
          test=self.test)
      return (run_idx,) + eval_outputs
    return self.fec.get_eval_outputs(
        model_spec=model_spec, run_idx=self.fixed_run_idx, test=self.test)
