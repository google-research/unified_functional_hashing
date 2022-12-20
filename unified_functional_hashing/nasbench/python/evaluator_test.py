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

"""Tests for evaluator."""

import copy
from unittest import mock

from absl.testing import absltest
from unified_functional_hashing.nasbench.python import constants
from unified_functional_hashing.nasbench.python import evaluator as evaluator_lib
from unified_functional_hashing.nasbench.python import hasher as hasher_lib
from unified_functional_hashing.nasbench.python import noise_generator as noise_generator_lib
from unified_functional_hashing.nasbench.python import test_util
from unified_functional_hashing.nasbench.python.testdata import computed_stats
import numpy as np

RNG_SEED = 42


class EvaluatorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    mock_nasbench = mock.MagicMock()
    mock_nasbench.get_metrics_from_spec = mock.MagicMock(
        side_effect=test_util.get_metrics)
    self._nasbench = mock_nasbench

    mock_model_spec = mock.MagicMock()
    mock_model_spec.original_matrix = np.array([[0, 1, 1, 1, 0, 1, 0],
                                                [0, 0, 0, 0, 0, 0, 1],
                                                [0, 0, 0, 0, 0, 0, 1],
                                                [0, 0, 0, 0, 1, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 1],
                                                [0, 0, 0, 0, 0, 0, 1],
                                                [0, 0, 0, 0, 0, 0, 0]])
    mock_model_spec.original_ops = [
        "input", "conv1x1-bn-relu", "conv3x3-bn-relu", "conv3x3-bn-relu",
        "conv3x3-bn-relu", "maxpool3x3", "output"
    ]
    mock_model_spec.matrix = copy.deepcopy(mock_model_spec.original_matrix)
    mock_model_spec.ops = copy.deepcopy(mock_model_spec.original_ops)
    mock_model_spec.data_format = "channels_last"
    mock_model_spec.graph_hash = "28cfc7874f6d200472e1a9dcd8650aa0"
    mock_model_spec.hash_spec = mock.MagicMock(
        return_value=mock_model_spec.graph_hash)
    self._model_spec = mock_model_spec

  def test_evaluate_different_fixed_run_indices(self):
    evaluator = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        fixed_run_idx=-1,
        mantissa_bits=24,
        hashing_time=0.0,
        noise_type="",
        noise_stddev=0.0,
        use_fec=False,
        max_num_evals=1,
        rng_seed=RNG_SEED)
    self.assertIsNone(evaluator.fec)
    graph_hash = self._model_spec.graph_hash
    stats = computed_stats.COMPUTED_STATS[graph_hash][108]

    for i in range(0, 3):
      evaluator.fixed_run_idx = i
      eval_outputs = evaluator.evaluate(model_spec=self._model_spec)
      run_idx, valid_acc, test_acc, train_time = eval_outputs
      self.assertEqual(run_idx, i)
      self.assertEqual(valid_acc, stats[i]["final_validation_accuracy"])
      self.assertEqual(test_acc, stats[i]["final_test_accuracy"])
      self.assertEqual(train_time, stats[i]["final_training_time"])

  def test_evaluate_no_fec(self):
    hashing_time = 10.0
    evaluator = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        mantissa_bits=24,
        hashing_time=hashing_time,
        noise_type="",
        noise_stddev=0.0,
        use_fec=False,
        max_num_evals=1,
        rng_seed=RNG_SEED)
    self.assertIsNone(evaluator.fec)
    graph_hash = self._model_spec.graph_hash
    stats = computed_stats.COMPUTED_STATS[graph_hash][108]
    eval_outputs = evaluator.evaluate(model_spec=self._model_spec)
    run_idx, valid_acc, test_acc, train_time = eval_outputs
    self.assertEqual(run_idx, 2)
    self.assertEqual(valid_acc, stats[2]["final_validation_accuracy"])
    self.assertEqual(test_acc, stats[2]["final_test_accuracy"])
    # No hashing_time added since no FEC.
    self.assertEqual(train_time, stats[2]["final_training_time"])

  def test_evaluate_no_fec_zero_noise(self):
    hashing_time = 10.0
    evaluator = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        mantissa_bits=24,
        hashing_time=hashing_time,
        noise_type="homoscedastic",
        noise_stddev=0.0,
        use_fec=False,
        max_num_evals=1,
        rng_seed=RNG_SEED)
    self.assertIsNone(evaluator.fec)
    graph_hash = self._model_spec.graph_hash
    stats = computed_stats.COMPUTED_STATS[graph_hash][108]
    eval_outputs = evaluator.evaluate(model_spec=self._model_spec)
    run_idx, valid_acc, test_acc, train_time = eval_outputs
    self.assertEqual(run_idx, 2)
    self.assertEqual(valid_acc, stats[2]["final_validation_accuracy"])
    self.assertEqual(test_acc, stats[2]["final_test_accuracy"])
    # No hashing_time added since no FEC.
    self.assertEqual(train_time, stats[2]["final_training_time"])

  def test_evaluate_no_fec_strong_noise(self):
    noise_type = "homoscedastic"
    noise_stddev = 0.1
    hashing_time = 10.0
    evaluator = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        mantissa_bits=24,
        hashing_time=hashing_time,
        noise_type=noise_type,
        noise_stddev=noise_stddev,
        use_fec=False,
        max_num_evals=1,
        rng_seed=RNG_SEED)
    self.assertIsNone(evaluator.fec)
    graph_hash = self._model_spec.graph_hash
    stats = computed_stats.COMPUTED_STATS[graph_hash][108]
    eval_outputs = evaluator.evaluate(model_spec=self._model_spec)
    run_idx, valid_acc, test_acc, train_time = eval_outputs
    rng = np.random.default_rng(RNG_SEED)
    noise_generator = noise_generator_lib.NoiseGenerator(
        noise_type=noise_type,
        noise_stddev=noise_stddev,
        rng_seed=rng.integers(low=1, high=constants.MAX_RNG_SEED, size=1)[0])
    noise = noise_generator.generate_noise()
    self.assertEqual(run_idx, 2)
    self.assertEqual(valid_acc, stats[2]["final_validation_accuracy"] + noise)
    self.assertEqual(test_acc, stats[2]["final_test_accuracy"])
    # No hashing_time added since no FEC.
    self.assertEqual(train_time, stats[2]["final_training_time"])

  def test_evaluate_fec_cache_miss(self):
    hashing_time = 10.0
    evaluator = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        mantissa_bits=24,
        hashing_time=hashing_time,
        noise_type="",
        noise_stddev=0.0,
        use_fec=True,
        max_num_evals=1,
        rng_seed=RNG_SEED)
    self.assertIsNotNone(evaluator.fec)
    graph_hash = self._model_spec.graph_hash
    stats = computed_stats.COMPUTED_STATS[graph_hash][108]
    eval_outputs = evaluator.evaluate(model_spec=self._model_spec)
    run_idx, valid_acc, test_acc, train_time = eval_outputs
    self.assertEqual(run_idx, 2)
    self.assertEqual(valid_acc, stats[2]["final_validation_accuracy"])
    self.assertEqual(test_acc, stats[2]["final_test_accuracy"])
    self.assertEqual(train_time, stats[2]["final_training_time"] + hashing_time)

  def test_evaluate_fec_cache_miss_zero_noise(self):
    hashing_time = 10.0
    evaluator = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        mantissa_bits=24,
        hashing_time=hashing_time,
        noise_type="homoscedastic",
        noise_stddev=0.0,
        use_fec=True,
        max_num_evals=1,
        rng_seed=RNG_SEED)
    self.assertIsNotNone(evaluator.fec)
    graph_hash = self._model_spec.graph_hash
    stats = computed_stats.COMPUTED_STATS[graph_hash][108]
    eval_outputs = evaluator.evaluate(model_spec=self._model_spec)
    run_idx, valid_acc, test_acc, train_time = eval_outputs
    self.assertEqual(run_idx, 2)
    self.assertEqual(valid_acc, stats[2]["final_validation_accuracy"])
    self.assertEqual(test_acc, stats[2]["final_test_accuracy"])
    self.assertEqual(train_time, stats[2]["final_training_time"] + hashing_time)

  def test_evaluate_fec_cache_miss_strong_noise(self):
    noise_type = "homoscedastic"
    noise_stddev = 0.1
    hashing_time = 10.0
    evaluator = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        mantissa_bits=24,
        hashing_time=hashing_time,
        noise_type=noise_type,
        noise_stddev=noise_stddev,
        use_fec=True,
        max_num_evals=1,
        rng_seed=RNG_SEED)
    self.assertIsNotNone(evaluator.fec)
    graph_hash = self._model_spec.graph_hash
    stats = computed_stats.COMPUTED_STATS[graph_hash][108]
    eval_outputs = evaluator.evaluate(model_spec=self._model_spec)
    run_idx, valid_acc, test_acc, train_time = eval_outputs
    rng = np.random.default_rng(RNG_SEED)
    noise_generator = noise_generator_lib.NoiseGenerator(
        noise_type=noise_type,
        noise_stddev=noise_stddev,
        rng_seed=rng.integers(low=1, high=constants.MAX_RNG_SEED, size=1)[0])
    noise = noise_generator.generate_noise()
    self.assertEqual(run_idx, 2)
    self.assertEqual(valid_acc, stats[2]["final_validation_accuracy"] + noise)
    self.assertEqual(test_acc, stats[2]["final_test_accuracy"])
    self.assertEqual(train_time, stats[2]["final_training_time"] + hashing_time)

  def test_evaluate_fec_cache_hit(self):
    hashing_time = 10.0
    evaluator = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        mantissa_bits=24,
        hashing_time=hashing_time,
        noise_type="",
        noise_stddev=0.0,
        use_fec=True,
        max_num_evals=1,
        rng_seed=RNG_SEED)
    self.assertIsNotNone(evaluator.fec)
    graph_hash = self._model_spec.graph_hash
    stats = computed_stats.COMPUTED_STATS[graph_hash][108]
    eval_outputs = evaluator.evaluate(model_spec=self._model_spec)
    run_idx, valid_acc, test_acc, train_time = eval_outputs
    self.assertEqual(run_idx, 2)
    self.assertEqual(valid_acc, stats[2]["final_validation_accuracy"])
    self.assertEqual(test_acc, stats[2]["final_test_accuracy"])
    self.assertEqual(train_time, stats[2]["final_training_time"] + hashing_time)

    # Again.
    eval_outputs = evaluator.evaluate(model_spec=self._model_spec)
    run_idx, valid_acc, test_acc, train_time = eval_outputs
    # Should be coming from computed_stats[108][2] in the cache.
    self.assertEqual(run_idx, 2)
    self.assertEqual(valid_acc, stats[2]["final_validation_accuracy"])
    self.assertEqual(test_acc, stats[2]["final_test_accuracy"])
    self.assertEqual(train_time, hashing_time)

  def test_evaluate_fec_standard_aggregation(self):
    hashing_time = 10.0
    evaluator = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        mantissa_bits=24,
        hashing_time=hashing_time,
        noise_type="",
        noise_stddev=0.0,
        use_fec=True,
        max_num_evals=2,
        rng_seed=RNG_SEED)
    self.assertIsNotNone(evaluator.fec)
    graph_hash = self._model_spec.graph_hash
    stats = computed_stats.COMPUTED_STATS[graph_hash][108]
    # Cache miss.
    eval_outputs = evaluator.evaluate(model_spec=self._model_spec)
    # From computed_stats[108][2].
    run_idx, valid_acc, test_acc, train_time = eval_outputs
    self.assertEqual(run_idx, 2)
    self.assertEqual(valid_acc, stats[2]["final_validation_accuracy"])
    self.assertEqual(test_acc, stats[2]["final_test_accuracy"])
    self.assertEqual(train_time, stats[2]["final_training_time"] + hashing_time)

    # Partial cache hit.
    eval_outputs = evaluator.evaluate(model_spec=self._model_spec)
    # mean(computed_stats[108][2], computed_stats[108][0]).
    run_idx, valid_acc, test_acc, training_time = eval_outputs
    self.assertEqual(run_idx, 0)
    self.assertAlmostEqual(
        valid_acc,
        np.mean([
            stats[2]["final_validation_accuracy"],
            stats[0]["final_validation_accuracy"]
        ]),
        delta=1e-8)
    self.assertAlmostEqual(
        test_acc,
        np.mean(
            [stats[2]["final_test_accuracy"], stats[0]["final_test_accuracy"]]),
        delta=1e-8)
    self.assertAlmostEqual(
        training_time,
        np.mean([
            stats[2]["final_training_time"], stats[0]["final_training_time"]
        ]) + hashing_time,
        delta=1e-8)

    # Full cache hit.
    eval_outputs = evaluator.evaluate(model_spec=self._model_spec)
    # mean(computed_stats[108][2], computed_stats[108][0]).
    run_idx, valid_acc, test_acc, training_time = eval_outputs
    self.assertEqual(run_idx, 0)
    self.assertAlmostEqual(
        valid_acc,
        np.mean([
            stats[2]["final_validation_accuracy"],
            stats[0]["final_validation_accuracy"]
        ]),
        delta=1e-8)
    self.assertAlmostEqual(
        test_acc,
        np.mean(
            [stats[2]["final_test_accuracy"], stats[0]["final_test_accuracy"]]),
        delta=1e-8)
    self.assertEqual(training_time, hashing_time)

  def test_evaluate_fec_sparse_aggregation(self):
    hashing_time = 10.0
    evaluator = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        mantissa_bits=24,
        hashing_time=hashing_time,
        noise_type="",
        noise_stddev=0.0,
        use_fec=True,
        max_num_evals=2,
        rng_seed=RNG_SEED)
    self.assertIsNotNone(evaluator.fec)
    graph_hash = self._model_spec.graph_hash
    stats = computed_stats.COMPUTED_STATS[graph_hash][108]
    # Cache miss.
    eval_outputs = evaluator.evaluate(model_spec=self._model_spec)
    # From computed_stats[108][2].
    run_idx, valid_acc, test_acc, train_time = eval_outputs
    self.assertEqual(run_idx, 2)
    self.assertEqual(valid_acc, stats[2]["final_validation_accuracy"])
    self.assertEqual(test_acc, stats[2]["final_test_accuracy"])
    self.assertEqual(train_time, stats[2]["final_training_time"] + hashing_time)

    # Partial cache hit.
    eval_outputs = evaluator.evaluate(model_spec=self._model_spec)
    # mean(computed_stats[108][2], computed_stats[108][0]).
    run_idx, valid_acc, test_acc, training_time = eval_outputs
    self.assertEqual(run_idx, 0)
    self.assertAlmostEqual(
        valid_acc,
        np.mean([
            stats[2]["final_validation_accuracy"],
            stats[0]["final_validation_accuracy"]
        ]),
        delta=1e-8)
    self.assertAlmostEqual(
        test_acc,
        np.mean(
            [stats[2]["final_test_accuracy"], stats[0]["final_test_accuracy"]]),
        delta=1e-8)
    self.assertAlmostEqual(
        training_time,
        np.mean([
            stats[2]["final_training_time"], stats[0]["final_training_time"]
        ]) + hashing_time,
        delta=1e-8)

    # Full cache hit.
    eval_outputs = evaluator.evaluate(model_spec=self._model_spec)
    # mean(computed_stats[108][2], computed_stats[108][0]).
    run_idx, valid_acc, test_acc, training_time = eval_outputs
    self.assertEqual(run_idx, 0)
    self.assertAlmostEqual(
        valid_acc,
        np.mean([
            stats[2]["final_validation_accuracy"],
            stats[0]["final_validation_accuracy"]
        ]),
        delta=1e-8)
    self.assertAlmostEqual(
        test_acc,
        np.mean(
            [stats[2]["final_test_accuracy"], stats[0]["final_test_accuracy"]]),
        delta=1e-8)
    self.assertEqual(training_time, hashing_time)

  def test_has_fec_true(self):
    evaluator = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        mantissa_bits=24,
        hashing_time=10.0,
        noise_type="",
        noise_stddev=0.0,
        use_fec=True,
        max_num_evals=2,
        rng_seed=RNG_SEED)
    self.assertTrue(evaluator.has_fec())

  def test_has_fec_false(self):
    evaluator = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        mantissa_bits=24,
        hashing_time=10.0,
        noise_type="",
        noise_stddev=0.0,
        use_fec=False,
        max_num_evals=2,
        rng_seed=RNG_SEED)
    self.assertFalse(evaluator.has_fec())

  def test_get_fec_statistics_cache_empty(self):
    evaluator = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        mantissa_bits=24,
        hashing_time=10.0,
        noise_type="",
        noise_stddev=0.0,
        use_fec=True,
        max_num_evals=1,
        rng_seed=RNG_SEED)
    self.assertIsNotNone(evaluator.fec)

    cache_stats = evaluator.get_fec_statistics()
    self.assertEqual(cache_stats["num_cache_misses"], 0)
    self.assertEqual(cache_stats["num_cache_partial_hits"], 0)
    self.assertEqual(cache_stats["num_cache_full_hits"], 0)
    self.assertEqual(cache_stats["cache_size"], 0)

  def test_get_fec_statistics_cache_with_history_empty(self):
    evaluator = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        mantissa_bits=24,
        hashing_time=10.0,
        noise_type="",
        noise_stddev=0.0,
        use_fec=True,
        max_num_evals=1,
        save_fec_history=True,
        rng_seed=RNG_SEED)
    self.assertIsNotNone(evaluator.fec)

    cache_stats = evaluator.get_fec_statistics()
    self.assertEqual(cache_stats["num_cache_misses"], 0)
    self.assertEqual(cache_stats["num_cache_partial_hits"], 0)
    self.assertEqual(cache_stats["num_cache_full_hits"], 0)
    self.assertEqual(cache_stats["cache_size"], 0)
    self.assertEqual(cache_stats["model_hashes"], "")

  def test_get_fec_statistics_no_cache_exception(self):
    evaluator = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        mantissa_bits=24,
        hashing_time=10.0,
        noise_type="",
        noise_stddev=0.0,
        use_fec=False,
        max_num_evals=1,
        rng_seed=RNG_SEED)
    self.assertIsNone(evaluator.fec)

    with self.assertRaisesRegex(
        expected_exception=AssertionError,
        expected_regex="Evaluator has no FEC!"):
      _ = evaluator.get_fec_statistics()

  def test_get_fec_statistics_cache_miss(self):
    evaluator = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        mantissa_bits=24,
        hashing_time=10.0,
        noise_type="",
        noise_stddev=0.0,
        use_fec=True,
        max_num_evals=1,
        rng_seed=RNG_SEED)
    self.assertIsNotNone(evaluator.fec)
    _ = evaluator.evaluate(model_spec=self._model_spec)

    cache_stats = evaluator.get_fec_statistics()
    self.assertEqual(cache_stats["num_cache_misses"], 1)
    self.assertEqual(cache_stats["num_cache_partial_hits"], 0)
    self.assertEqual(cache_stats["num_cache_full_hits"], 0)
    self.assertEqual(cache_stats["cache_size"], 1)

  def test_get_fec_statistics_cache_hit(self):
    evaluator = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        mantissa_bits=24,
        hashing_time=10.0,
        noise_type="",
        noise_stddev=0.0,
        use_fec=True,
        max_num_evals=1,
        rng_seed=RNG_SEED)
    self.assertIsNotNone(evaluator.fec)

    # Cache miss.
    _ = evaluator.evaluate(model_spec=self._model_spec)

    cache_stats = evaluator.get_fec_statistics()
    self.assertEqual(cache_stats["num_cache_misses"], 1)
    self.assertEqual(cache_stats["num_cache_partial_hits"], 0)
    self.assertEqual(cache_stats["num_cache_full_hits"], 0)
    self.assertEqual(cache_stats["cache_size"], 1)

    # Cache hit.
    _ = evaluator.evaluate(model_spec=self._model_spec)

    cache_stats = evaluator.get_fec_statistics()
    self.assertEqual(cache_stats["num_cache_misses"], 1)
    self.assertEqual(cache_stats["num_cache_partial_hits"], 0)
    self.assertEqual(cache_stats["num_cache_full_hits"], 1)
    self.assertEqual(cache_stats["cache_size"], 1)

  def test_get_fec_statistics_cache_partial_hit(self):
    evaluator = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        mantissa_bits=24,
        hashing_time=10.0,
        noise_type="",
        noise_stddev=0.0,
        use_fec=True,
        max_num_evals=2,
        rng_seed=RNG_SEED)
    self.assertIsNotNone(evaluator.fec)

    # Cache miss.
    self._model_spec.graph_hash = "28cfc7874f6d200472e1a9dcd8650aa0"
    _ = evaluator.evaluate(model_spec=self._model_spec)

    cache_stats = evaluator.get_fec_statistics()
    self.assertEqual(cache_stats["num_cache_misses"], 1)
    self.assertEqual(cache_stats["num_cache_partial_hits"], 0)
    self.assertEqual(cache_stats["num_cache_full_hits"], 0)
    self.assertEqual(cache_stats["cache_size"], 1)

    # Cache partial hit.
    _ = evaluator.evaluate(model_spec=self._model_spec)

    cache_stats = evaluator.get_fec_statistics()
    self.assertEqual(cache_stats["num_cache_misses"], 1)
    self.assertEqual(cache_stats["num_cache_partial_hits"], 1)
    self.assertEqual(cache_stats["num_cache_full_hits"], 0)
    self.assertEqual(cache_stats["cache_size"], 1)

  def test_get_fec_statistics_cache_full_hit(self):
    evaluator = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        mantissa_bits=24,
        hashing_time=10.0,
        noise_type="",
        noise_stddev=0.0,
        use_fec=True,
        max_num_evals=2,
        rng_seed=RNG_SEED)
    self.assertIsNotNone(evaluator.fec)

    # Cache miss.
    self._model_spec.graph_hash = "28cfc7874f6d200472e1a9dcd8650aa0"
    _ = evaluator.evaluate(model_spec=self._model_spec)

    cache_stats = evaluator.get_fec_statistics()
    self.assertEqual(cache_stats["num_cache_misses"], 1)
    self.assertEqual(cache_stats["num_cache_partial_hits"], 0)
    self.assertEqual(cache_stats["num_cache_full_hits"], 0)
    self.assertEqual(cache_stats["cache_size"], 1)

    # Cache partial hit.
    _ = evaluator.evaluate(model_spec=self._model_spec)

    cache_stats = evaluator.get_fec_statistics()
    self.assertEqual(cache_stats["num_cache_misses"], 1)
    self.assertEqual(cache_stats["num_cache_partial_hits"], 1)
    self.assertEqual(cache_stats["num_cache_full_hits"], 0)
    self.assertEqual(cache_stats["cache_size"], 1)

    # Cache full hit.
    _ = evaluator.evaluate(model_spec=self._model_spec)

    cache_stats = evaluator.get_fec_statistics()
    self.assertEqual(cache_stats["num_cache_misses"], 1)
    self.assertEqual(cache_stats["num_cache_partial_hits"], 1)
    self.assertEqual(cache_stats["num_cache_full_hits"], 1)
    self.assertEqual(cache_stats["cache_size"], 1)

  def test_get_fec_statistics_cache_hit_forgotten_hash(self):
    evaluator = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        mantissa_bits=24,
        hashing_time=10.0,
        noise_type="",
        noise_stddev=0.0,
        use_fec=True,
        max_num_evals=1,
        fec_remove_probability=1.0,
        rng_seed=RNG_SEED)
    self.assertIsNotNone(evaluator.fec)

    # Cache miss.
    _ = evaluator.evaluate(model_spec=self._model_spec)

    cache_stats = evaluator.get_fec_statistics()
    self.assertEqual(cache_stats["num_cache_misses"], 1)
    self.assertEqual(cache_stats["num_cache_partial_hits"], 0)
    self.assertEqual(cache_stats["num_cache_full_hits"], 0)
    self.assertEqual(cache_stats["cache_size"], 1)

    # Cache hit.
    _ = evaluator.evaluate(model_spec=self._model_spec)

    cache_stats = evaluator.get_fec_statistics()
    self.assertEqual(cache_stats["num_cache_misses"], 1)
    self.assertEqual(cache_stats["num_cache_partial_hits"], 0)
    self.assertEqual(cache_stats["num_cache_full_hits"], 1)
    self.assertEqual(cache_stats["cache_size"], 0)

  def test_get_fec_statistics_multiple_cache_hits_interleaved(self):
    evaluator = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        mantissa_bits=24,
        hashing_time=10.0,
        noise_type="",
        noise_stddev=0.0,
        use_fec=True,
        max_num_evals=1,
        fec_remove_probability=0.0,
        rng_seed=RNG_SEED)
    self.assertIsNotNone(evaluator.fec)

    # Should be fresh.
    cache_stats = evaluator.get_fec_statistics()
    self.assertEqual(cache_stats["num_cache_misses"], 0)
    self.assertEqual(cache_stats["num_cache_partial_hits"], 0)
    self.assertEqual(cache_stats["num_cache_full_hits"], 0)
    self.assertEqual(cache_stats["cache_size"], 0)

    # Model A cache miss.
    self._model_spec.graph_hash = "28cfc7874f6d200472e1a9dcd8650aa0"
    _ = evaluator.evaluate(model_spec=self._model_spec)

    cache_stats = evaluator.get_fec_statistics()
    self.assertEqual(cache_stats["num_cache_misses"], 1)
    self.assertEqual(cache_stats["num_cache_partial_hits"], 0)
    self.assertEqual(cache_stats["num_cache_full_hits"], 0)
    self.assertEqual(cache_stats["cache_size"], 1)

    # Model B cache miss.
    self._model_spec.graph_hash = "1"
    _ = evaluator.evaluate(model_spec=self._model_spec)

    cache_stats = evaluator.get_fec_statistics()
    self.assertEqual(cache_stats["num_cache_misses"], 2)
    self.assertEqual(cache_stats["num_cache_partial_hits"], 0)
    self.assertEqual(cache_stats["num_cache_full_hits"], 0)
    self.assertEqual(cache_stats["cache_size"], 2)

    # Model C cache miss.
    self._model_spec.graph_hash = "2"
    _ = evaluator.evaluate(model_spec=self._model_spec)

    cache_stats = evaluator.get_fec_statistics()
    self.assertEqual(cache_stats["num_cache_misses"], 3)
    self.assertEqual(cache_stats["num_cache_partial_hits"], 0)
    self.assertEqual(cache_stats["num_cache_full_hits"], 0)
    self.assertEqual(cache_stats["cache_size"], 3)

    # Model A again, now cache hit.
    self._model_spec.graph_hash = "28cfc7874f6d200472e1a9dcd8650aa0"
    _ = evaluator.evaluate(model_spec=self._model_spec)

    cache_stats = evaluator.get_fec_statistics()
    self.assertEqual(cache_stats["num_cache_misses"], 3)
    self.assertEqual(cache_stats["num_cache_partial_hits"], 0)
    self.assertEqual(cache_stats["num_cache_full_hits"], 1)
    self.assertEqual(cache_stats["cache_size"], 3)

    # Model C again, now cache hit.
    self._model_spec.graph_hash = "2"
    _ = evaluator.evaluate(model_spec=self._model_spec)

    cache_stats = evaluator.get_fec_statistics()
    self.assertEqual(cache_stats["num_cache_misses"], 3)
    self.assertEqual(cache_stats["num_cache_partial_hits"], 0)
    self.assertEqual(cache_stats["num_cache_full_hits"], 2)
    self.assertEqual(cache_stats["cache_size"], 3)

  def test_get_fec_statistics_hashes_correctly(self):
    mantissa_bits = 24
    hashing_time = 10.0
    evaluator = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        mantissa_bits=mantissa_bits,
        hashing_time=hashing_time,
        noise_type="",
        noise_stddev=0.0,
        use_fec=True,
        max_num_evals=1,
        save_fec_history=True,
        rng_seed=RNG_SEED)
    self.assertIsNotNone(evaluator.fec)

    _ = evaluator.evaluate(model_spec=self._model_spec)
    cache_stats = evaluator.get_fec_statistics()

    hasher = hasher_lib.Hasher(
        nasbench=self._nasbench,
        mantissa_bits=mantissa_bits,
        hashing_time=hashing_time)
    expected_model_hash, _ = hasher.get_unified_functional_hash(
        self._model_spec)

    self.assertEqual(cache_stats["num_cache_misses"], 1)
    self.assertEqual(cache_stats["num_cache_partial_hits"], 0)
    self.assertEqual(cache_stats["num_cache_full_hits"], 0)
    self.assertEqual(cache_stats["cache_size"], 1)
    self.assertEqual(cache_stats["model_hashes"], str(expected_model_hash))


if __name__ == "__main__":
  absltest.main()
