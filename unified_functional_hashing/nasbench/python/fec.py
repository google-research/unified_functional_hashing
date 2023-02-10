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

"""Functional Equivalence Cache for NASBench."""

from typing import Dict, List, Optional, Tuple, Union

from unified_functional_hashing.nasbench.python import evaluation
from unified_functional_hashing.nasbench.python import hasher as hasher_lib
from unified_functional_hashing.nasbench.python import noise_generator as noise_generator_lib
from nasbench import api
import numpy as np


class FEC():
  """Functional Equivalence Cache."""

  def __init__(self,
               nasbench: api.NASBench,
               noise_generator: noise_generator_lib.NoiseGenerator,
               mantissa_bits: int = 24,
               hashing_time: float = 10.0,
               max_num_evals: int = 1,
               remove_probability: float = 0.0,
               save_history: bool = False,
               rng_seed: Optional[int] = None):
    """Initializes FEC.

    Args:
      nasbench: NASBench instance.
      noise_generator: NoiseGenerator instance.
      mantissa_bits: number of bits to use in the mantissa.
      hashing_time: Number of seconds it takes to generate hash.
      max_num_evals: The maximum number of evaluations to perform for a hash.
      remove_probability: After a cache hit, the probability a key will be
        removed from the cache.
      save_history: Whether to save history or not. If True, the FEC's number of
        cache misses, cache partial hits, cache full hits, cache size, and the
        Hasher's 4 accuracy model hash will be saved.
      rng_seed: Seed to initialize rng.
    """
    self.nasbench = nasbench
    self.rng = np.random.default_rng(rng_seed)
    self.max_num_evals = max_num_evals
    self.remove_probability = remove_probability
    self.save_history = save_history
    self.cache = dict()
    self.hasher = hasher_lib.Hasher(nasbench, mantissa_bits, hashing_time)

    self.noise_generator = noise_generator
    self.run_idx_set = {0, 1, 2}
    self.num_cache_misses = 0
    self.num_cache_partial_hits = 0
    self.num_cache_full_hits = 0
    self.most_recent_model_hash = ""

  def get_cache_statistics(self) -> Dict[str, Union[int, str]]:
    """Gets cache statistics.

    Returns:
      Dict of the number of cache misses, partial hits, and full hits and the
        current size of the cache.
    """
    return {
        "num_cache_misses": self.num_cache_misses,
        "num_cache_partial_hits": self.num_cache_partial_hits,
        "num_cache_full_hits": self.num_cache_full_hits,
        "cache_size": len(self.cache),
        "model_hashes": self.most_recent_model_hash
    }

  def init_new_hash(self, model_hash: int) -> None:
    """Initializes new hash in cache.

    Args:
      model_hash: integer hash of model spec.
    """
    self.cache[model_hash] = {
        "run_idx": -1,
        "valid_accuracy": 0.0,
        "test_accuracy": 0.0,
        "training_time": 0.0,
        "num_evals": 0
    }

  def update_cache(self,
                   model_hash: int,
                   model_spec: api.ModelSpec,
                   run_idx: int = -1,
                   test: bool = False) -> None:
    """Updates cache at hash for model spec.

    Args:
      model_hash: integer hash of model spec.
      model_spec: ModelSpec matrix.
      run_idx: The run index to use for evaluation.
      test: Whether called from a test or not.
    """
    if run_idx not in self.run_idx_set:
      run_idx = self.rng.integers(low=0, high=3, size=1)[0]

    eval_outputs = evaluation.evaluate(
        nasbench=self.nasbench,
        noise_generator=self.noise_generator,
        model_spec=model_spec,
        run_idx=run_idx,
        test=test)
    num_evals = self.cache[model_hash]["num_evals"]
    for i, k in enumerate(["valid_accuracy", "test_accuracy", "training_time"]):
      val = self.cache[model_hash][k]
      new_val = (val * num_evals + eval_outputs[i]) / (num_evals + 1)
      self.cache[model_hash][k] = new_val
    self.cache[model_hash]["run_idx"] = run_idx
    self.cache[model_hash]["num_evals"] += 1

  def filter_cache_hits(self, keys: List[int]) -> List[int]:
    """Filters cache hits occasionally.

    This prevents cache hits that store high evaluation outliers from having a
    large impact in the search process.

    Args:
      keys: List of removal candidate hashes.

    Returns:
      List of keys to remove.
    """
    keys_to_remove = []
    for key in keys:
      if self.rng.random(size=1)[0] < self.remove_probability:
        keys_to_remove.append(key)
    return keys_to_remove

  def remove_fitnesses_in_cache(self, keys: List[int]) -> None:
    """Removes fitnesses in cache.

    Args:
      keys: List of keys to remove.
    """
    for key in keys:
      self.cache.pop(key)

  def get_eval_outputs(self,
                       model_spec: api.ModelSpec,
                       run_idx: int = -1,
                       test: bool = False) -> Tuple[int, float, float, float]:
    """Gets eval outputs from FEC based on hash of ModelSpec.

    Args:
      model_spec: ModelSpec matrix.
      run_idx: The run index to use for evaluation.
      test: Whether called from a test or not.

    Returns:
      4-tuple of int run_idx and floats for valid and test accuracies and
        training time.
    """
    # A hash costs a relatively small amount of time.
    model_hash, hashing_time = self.hasher.get_unified_functional_hash(
        model_spec, test)
    self.most_recent_model_hash = str(model_hash)
    if model_hash not in self.cache:
      #  Cache miss.
      self.num_cache_misses += 1
      self.init_new_hash(model_hash)
    elif self.cache[model_hash]["num_evals"] < self.max_num_evals:
      # Cache partial hit.
      self.num_cache_partial_hits += 1

    removal_candidates = []
    if self.cache[model_hash]["num_evals"] < self.max_num_evals:
      # Cache partial hit.
      self.update_cache(model_hash, model_spec, run_idx, test)
      eval_outputs = (int(self.cache[model_hash]["run_idx"]),
                      self.cache[model_hash]["valid_accuracy"],
                      self.cache[model_hash]["test_accuracy"],
                      self.cache[model_hash]["training_time"] + hashing_time)
    else:
      # Full cache hit, purely a lookup.
      self.num_cache_full_hits += 1
      eval_outputs = (int(self.cache[model_hash]["run_idx"]),
                      self.cache[model_hash]["valid_accuracy"],
                      self.cache[model_hash]["test_accuracy"], hashing_time)
      removal_candidates.append(model_hash)

    if self.remove_probability > 0.0:
      keys_to_remove = self.filter_cache_hits(removal_candidates)
      self.remove_fitnesses_in_cache(keys_to_remove)

    return eval_outputs
