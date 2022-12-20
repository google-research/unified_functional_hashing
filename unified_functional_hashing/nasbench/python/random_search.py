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

"""Random searcher for NASBench.
"""

from typing import Dict, List, Optional, Union

from unified_functional_hashing.nasbench.python import constants
from unified_functional_hashing.nasbench.python import evaluator as evaluator_lib
from unified_functional_hashing.nasbench.python import random_spec_generator as random_spec_generator_lib
from unified_functional_hashing.nasbench.python import util
from nasbench import api
import numpy as np


class RandomSearcher():
  """RandomSearcher for finding best ModelSpecs through random search.
  """

  def __init__(
      self,
      nasbench: api.NASBench,
      evaluator: evaluator_lib.Evaluator,
      max_time_budget: float = 5e6,
      save_child_history: bool = False,
      rng_seed: Optional[int] = None,
      statistics: Optional[Dict[str, List[Union[float, int, str]]]] = None):
    """Initializes RandomSearcher.

    Args:
      nasbench: NASBench instance.
      evaluator: Evaluator to calculate valid and test accuracies.
      max_time_budget: Maximum time budget for search.
      save_child_history: Whether to save history of child. If True, its
        isomorphism-invariant graph hash, run indices, validation & test
        accuracies, and training times will be saved.
      rng_seed: Seed to initialize rng.
      statistics: Dict to track searcher's statistics.
    """
    self.nasbench = nasbench
    self.evaluator = evaluator
    self.rng = np.random.default_rng(rng_seed)
    self.max_time_budget = max_time_budget
    self.save_child_history = save_child_history
    self.random_spec_generator = random_spec_generator_lib.RandomSpecGenerator(
        nasbench,
        rng_seed=self.rng.integers(
            low=1, high=constants.MAX_RNG_SEED, size=1)[0])
    self.time_spent = 0.0
    if statistics is None:
      self.statistics = self.initialze_statistics()
    else:
      self.statistics = statistics

  def initialze_statistics(self) -> Dict[str, List[Union[float, int, str]]]:
    """Initializes searcher's statistics.

    Returns:
      Dict of statistic lists.
    """
    statistics = {"times": [0.0], "best_valids": [0.0], "best_tests": [0.0]}
    if self.evaluator.has_fec() and self.evaluator.fec.save_history:
      statistics.update({
          "num_cache_misses": [0],
          "num_cache_partial_hits": [0],
          "num_cache_full_hits": [0],
          "cache_size": [0],
          # Model hash is computed by a Hasher using 4 accuracies from the
          # computed_stats of a ModelSpec
          "model_hashes": [""]})

    if self.save_child_history:
      statistics.update({
          "run_indices": [-1],
          "validation_accuracies": [0.0],
          "test_accuracies": [0.0],
          "training_times": [0.0],
          # Graph hash is the isomorphism-invariant graph hash of a ModelSpec.
          "graph_hashes": [""]})
    return statistics

  def get_statistics(self) -> Dict[str, List[Union[float, int, str]]]:
    """Gets searcher's statistics.

    Returns:
      Dict of statistic lists.
    """
    return self.statistics

  def perform_search_iteration(self) -> None:
    """Performs a single iteration of random search.
    """
    spec = self.random_spec_generator.random_spec()
    eval_outputs = self.evaluator.evaluate(spec)
    self.time_spent += eval_outputs[-1]
    self.statistics["times"].append(self.time_spent)

    # It's important to select models only based on validation accuracy, test
    # accuracy is used only for comparing different search trajectories.
    util.update_best_fitnesses(
        (eval_outputs[1], eval_outputs[2]),
        self.statistics["best_valids"],
        self.statistics["best_tests"])

    if self.evaluator.has_fec() and self.evaluator.fec.save_history:
      fec_stats = self.evaluator.get_fec_statistics()
      for k, v in fec_stats.items():
        if k in self.statistics:
          self.statistics[k].append(v)

    if self.save_child_history:
      self.statistics["run_indices"].append(eval_outputs[0])
      self.statistics["validation_accuracies"].append(eval_outputs[1])
      self.statistics["test_accuracies"].append(eval_outputs[2])
      self.statistics["training_times"].append(eval_outputs[3])
      self.statistics["graph_hashes"].append(
          spec.hash_spec(
              canonical_ops=self.nasbench.config["nasbench_available_ops"]))

  def run_search(self) -> None:
    """Runs a single roll-out of random search to a fixed time budget.
    """
    iterations = 0
    while self.time_spent < self.max_time_budget:
      self.perform_search_iteration()

      iterations += 1
      if self.time_spent >= self.max_time_budget:
        print("iterations = {}".format(iterations))
