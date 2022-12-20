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

"""Tests for test_util."""

from unittest import mock

from absl.testing import absltest
from unified_functional_hashing.nasbench.python import test_util
from unified_functional_hashing.nasbench.python.testdata import computed_stats as computed_stats_lib
import numpy as np


RNG_SEED = 42


class TestUtilTest(absltest.TestCase):

  def test_get_metrics(self):
    mock_model_spec = mock.MagicMock()
    mock_model_spec.graph_hash = "e6a21c47208f1f9f3f954887665520a8"
    fixed_stats, computed_stats = test_util.get_metrics(
        model_spec=mock_model_spec)
    self.assertIsNone(fixed_stats)
    self.assertDictEqual(
        computed_stats,
        computed_stats_lib.COMPUTED_STATS["e6a21c47208f1f9f3f954887665520a8"])

  def test_update_expected_iterators_no_fec(self):
    rng = np.random.default_rng(RNG_SEED)
    times, best_valids, best_tests = [0.0], [0.0], [0.0]
    test_util.update_expected_iterators(
        expected_rng=rng,
        times=times,
        best_valids=best_valids,
        best_tests=best_tests,
        graph_hash="e6a21c47208f1f9f3f954887665520a8",
        use_fec=False)
    computed_stats = computed_stats_lib.COMPUTED_STATS
    hashed_stats = computed_stats["e6a21c47208f1f9f3f954887665520a8"]
    stats = hashed_stats[108][0]
    expected_times = [0.0, stats["final_training_time"]]
    expected_best_valids = [0.0, stats["final_validation_accuracy"]]
    expected_best_tests = [0.0, stats["final_test_accuracy"]]
    self.assertListEqual(times, expected_times)
    self.assertListEqual(best_valids, expected_best_valids)
    self.assertListEqual(best_tests, expected_best_tests)

  def test_update_expected_iterators_fec_cache_hit_false(self):
    rng = np.random.default_rng(RNG_SEED)
    times, best_valids, best_tests = [0.0], [0.0], [0.0]
    hashing_time = 10.0
    test_util.update_expected_iterators(
        expected_rng=rng,
        times=times,
        best_valids=best_valids,
        best_tests=best_tests,
        graph_hash="e6a21c47208f1f9f3f954887665520a8",
        use_fec=True,
        cache_hit=False,
        hashing_time=hashing_time)
    computed_stats = computed_stats_lib.COMPUTED_STATS
    hashed_stats = computed_stats["e6a21c47208f1f9f3f954887665520a8"]
    stats = hashed_stats[108][0]
    expected_times = [0.0, stats["final_training_time"] + hashing_time]
    expected_best_valids = [0.0, stats["final_validation_accuracy"]]
    expected_best_tests = [0.0, stats["final_test_accuracy"]]
    self.assertListEqual(times, expected_times)
    self.assertListEqual(best_valids, expected_best_valids)
    self.assertListEqual(best_tests, expected_best_tests)

  def test_update_expected_iterators_fec_cache_hit_true(self):
    rng = np.random.default_rng(RNG_SEED)
    times, best_valids, best_tests = [0.0], [0.0], [0.0]
    hashing_time = 10.0
    test_util.update_expected_iterators(
        expected_rng=rng,
        times=times,
        best_valids=best_valids,
        best_tests=best_tests,
        graph_hash="e6a21c47208f1f9f3f954887665520a8",
        use_fec=True,
        cache_hit=True,
        hashing_time=hashing_time)
    computed_stats = computed_stats_lib.COMPUTED_STATS
    hashed_stats = computed_stats["e6a21c47208f1f9f3f954887665520a8"]
    stats = hashed_stats[108][0]
    expected_times = [0.0, hashing_time]
    expected_best_valids = [0.0, stats["final_validation_accuracy"]]
    expected_best_tests = [0.0, stats["final_test_accuracy"]]
    self.assertListEqual(times, expected_times)
    self.assertListEqual(best_valids, expected_best_valids)
    self.assertListEqual(best_tests, expected_best_tests)


if __name__ == "__main__":
  absltest.main()
