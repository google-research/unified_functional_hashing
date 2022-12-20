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

"""Test utilities for NASBench.
"""

from typing import Dict, List, Tuple
from unittest import mock

from unified_functional_hashing.nasbench.python import util
from unified_functional_hashing.nasbench.python.testdata import computed_stats
import numpy as np


def get_metrics(
    model_spec: mock.MagicMock
    ) -> Tuple[None, Dict[int, List[Dict[str, float]]]]:
  model_hash = model_spec.graph_hash
  return (None, computed_stats.COMPUTED_STATS[model_hash])


def update_expected_iterators(
    expected_rng: np.random.default_rng,
    times: List[float],
    best_valids: List[float],
    best_tests: List[float],
    graph_hash: str,
    use_fec: bool = False,
    cache_hit: bool = False,
    hashing_time: float = 10.0) -> None:
  """Updates expected iterators.

  Args:
    expected_rng: Numpy RNG that has the expected behavior of the one used.
    times: List of elapsed training times for each iteration.
    best_valids: List of best validation fitnesses for each iteration.
    best_tests: List of best test fitnesses for each iteration.
    graph_hash: String hash of model graph.
    use_fec: Whether FEC was used or not.
    cache_hit: Whether a cache hit or not.
    hashing_time: Number of seconds it takes to generate hash.
  """
  expected_run_idx = expected_rng.integers(low=0, high=3, size=1)[0]
  stats = computed_stats.COMPUTED_STATS[graph_hash][108][expected_run_idx]
  new_time = times[-1]
  if use_fec:
    new_time += hashing_time
  if not cache_hit:
    new_time += stats["final_training_time"]
  times.append(new_time)
  fitnesses = (stats["final_validation_accuracy"],
               stats["final_test_accuracy"])
  util.update_best_fitnesses(
      fitnesses=fitnesses, best_valids=best_valids, best_tests=best_tests)
