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

"""Tests for hasher."""

from unittest import mock

from absl.testing import absltest
from unified_functional_hashing.nasbench.python import hasher as hasher_lib
from unified_functional_hashing.nasbench.python import test_util
from unified_functional_hashing.nasbench.python.testdata import computed_stats


class HasherTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    mock_nasbench = mock.MagicMock()
    mock_nasbench.get_metrics_from_spec = mock.MagicMock(
        side_effect=test_util.get_metrics)
    self._nasbench = mock_nasbench

    mock_model_spec = mock.MagicMock()
    mock_model_spec.graph_hash = "28cfc7874f6d200472e1a9dcd8650aa0"
    self._model_spec = mock_model_spec

    mock_model_spec2 = mock.MagicMock()
    mock_model_spec2.graph_hash = "1"
    self._model_spec2 = mock_model_spec2

  def test_hash_uses_correct_computed_stats(self):
    mantissa_bits = 24
    hasher = hasher_lib.Hasher(self._nasbench, mantissa_bits=mantissa_bits)
    model_hash, _ = hasher.get_unified_functional_hash(self._model_spec)
    stats = computed_stats.COMPUTED_STATS
    hashed_stats = stats["28cfc7874f6d200472e1a9dcd8650aa0"]
    expected_hash = hasher.significant_float_mix([
        hashed_stats[4][0]["final_train_accuracy"],
        hashed_stats[4][0]["halfway_train_accuracy"],
        hashed_stats[4][0]["final_validation_accuracy"],
        hashed_stats[4][0]["halfway_validation_accuracy"]
    ], mantissa_bits)
    self.assertEqual(model_hash, expected_hash)

  def test_correct_hashing_time(self):
    expected_hashing_time = 10.0
    hasher = hasher_lib.Hasher(
        self._nasbench, mantissa_bits=24, hashing_time=expected_hashing_time)
    _, hashing_time = hasher.get_unified_functional_hash(self._model_spec)
    self.assertEqual(hashing_time, expected_hashing_time)

  def test_hash_matches_hardcoded(self):
    hasher = hasher_lib.Hasher(
        self._nasbench, mantissa_bits=24, hashing_time=10.0)
    model_hash, _ = hasher.get_unified_functional_hash(self._model_spec)
    self.assertEqual(model_hash, 6559424274240569344)

  def test_same_hash_from_same_spec(self):
    hasher = hasher_lib.Hasher(
        self._nasbench, mantissa_bits=24, hashing_time=10.0)
    model_hash1, _ = hasher.get_unified_functional_hash(self._model_spec)
    model_hash2, _ = hasher.get_unified_functional_hash(self._model_spec)
    self.assertEqual(model_hash1, model_hash2)

  def test_different_mantissa_bits(self):
    hasher1 = hasher_lib.Hasher(
        self._nasbench, mantissa_bits=24, hashing_time=10.0)
    model_hash1, _ = hasher1.get_unified_functional_hash(self._model_spec)
    hasher2 = hasher_lib.Hasher(
        self._nasbench, mantissa_bits=8, hashing_time=10.0)
    model_hash2, _ = hasher2.get_unified_functional_hash(self._model_spec)
    self.assertNotEqual(model_hash1, model_hash2)

  def test_different_computed_stats_hashing_sensitivity(self):
    hasher = hasher_lib.Hasher(
        self._nasbench, mantissa_bits=24, hashing_time=10.0)
    model_hash1, _ = hasher.get_unified_functional_hash(self._model_spec)
    model_hash2, _ = hasher.get_unified_functional_hash(self._model_spec2)
    self.assertNotEqual(model_hash1, model_hash2)


if __name__ == "__main__":
  absltest.main()
