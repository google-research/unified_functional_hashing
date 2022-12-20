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

"""Tests for evaluation."""

from unittest import mock

from absl.testing import absltest
from unified_functional_hashing.nasbench.python import evaluation
from unified_functional_hashing.nasbench.python import noise_generator as noise_generator_lib
from unified_functional_hashing.nasbench.python import test_util
from unified_functional_hashing.nasbench.python.testdata import computed_stats


RNG_SEED = 42


class EvaluationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    mock_nasbench = mock.MagicMock()
    mock_nasbench.get_metrics_from_spec = mock.MagicMock(
        side_effect=test_util.get_metrics)
    self._nasbench = mock_nasbench

    mock_model_spec = mock.MagicMock()
    mock_model_spec.graph_hash = "28cfc7874f6d200472e1a9dcd8650aa0"
    self._model_spec = mock_model_spec

  def test_evaluate_eval_outputs_format(self):
    noise_generator = noise_generator_lib.NoiseGenerator(
        noise_type="", noise_stddev=0.0, rng_seed=RNG_SEED)
    eval_outputs = evaluation.evaluate(
        nasbench=self._nasbench,
        noise_generator=noise_generator,
        model_spec=self._model_spec,
        run_idx=0
    )
    self.assertIsInstance(eval_outputs, tuple)
    self.assertLen(eval_outputs, 3)

  def test_evaluate_no_noise_all_runs(self):
    noise_generator = noise_generator_lib.NoiseGenerator(
        noise_type="", noise_stddev=0.0, rng_seed=RNG_SEED)
    # Run 0.
    eval_outputs = evaluation.evaluate(
        nasbench=self._nasbench,
        noise_generator=noise_generator,
        model_spec=self._model_spec,
        run_idx=0
    )
    valid_acc, test_acc, train_time = eval_outputs
    stats = computed_stats.COMPUTED_STATS[self._model_spec.graph_hash][108][0]
    self.assertEqual(valid_acc, stats["final_validation_accuracy"])
    self.assertEqual(test_acc, stats["final_test_accuracy"])
    self.assertEqual(train_time, stats["final_training_time"])

    # Run 1.
    eval_outputs = evaluation.evaluate(
        nasbench=self._nasbench,
        noise_generator=noise_generator,
        model_spec=self._model_spec,
        run_idx=1
    )
    valid_acc, test_acc, train_time = eval_outputs
    stats = computed_stats.COMPUTED_STATS[self._model_spec.graph_hash][108][1]
    self.assertEqual(valid_acc, stats["final_validation_accuracy"])
    self.assertEqual(test_acc, stats["final_test_accuracy"])
    self.assertEqual(train_time, stats["final_training_time"])

    # Run 2.
    eval_outputs = evaluation.evaluate(
        nasbench=self._nasbench,
        noise_generator=noise_generator,
        model_spec=self._model_spec,
        run_idx=2
    )
    valid_acc, test_acc, train_time = eval_outputs
    stats = computed_stats.COMPUTED_STATS[self._model_spec.graph_hash][108][2]
    self.assertEqual(valid_acc, stats["final_validation_accuracy"])
    self.assertEqual(test_acc, stats["final_test_accuracy"])
    self.assertEqual(train_time, stats["final_training_time"])

  def test_evaluate_zero_homoscedastic_noise(self):
    noise_generator = noise_generator_lib.NoiseGenerator(
        noise_type="homoscedastic", noise_stddev=0.0, rng_seed=RNG_SEED)
    eval_outputs = evaluation.evaluate(
        nasbench=self._nasbench,
        noise_generator=noise_generator,
        model_spec=self._model_spec,
        run_idx=0
    )
    valid_acc, test_acc, train_time = eval_outputs
    stats = computed_stats.COMPUTED_STATS[self._model_spec.graph_hash][108][0]
    self.assertEqual(valid_acc, stats["final_validation_accuracy"])
    self.assertEqual(test_acc, stats["final_test_accuracy"])
    self.assertEqual(train_time, stats["final_training_time"])

  def test_evaluate_strong_homoscedastic_noise(self):
    noise_generator = noise_generator_lib.NoiseGenerator(
        noise_type="homoscedastic", noise_stddev=0.1, rng_seed=RNG_SEED)
    eval_outputs = evaluation.evaluate(
        nasbench=self._nasbench,
        noise_generator=noise_generator,
        model_spec=self._model_spec,
        run_idx=0
    )
    valid_acc, test_acc, train_time = eval_outputs
    expected_noise_generator = noise_generator_lib.NoiseGenerator(
        noise_type="homoscedastic", noise_stddev=0.1, rng_seed=RNG_SEED)
    noise = expected_noise_generator.generate_noise()
    stats = computed_stats.COMPUTED_STATS[self._model_spec.graph_hash][108][0]
    self.assertEqual(valid_acc, stats["final_validation_accuracy"] + noise)
    self.assertEqual(test_acc, stats["final_test_accuracy"])
    self.assertEqual(train_time, stats["final_training_time"])

  def test_evaluate_strong_homoscedastic_noise_reproducible_seed(self):
    noise_generator = noise_generator_lib.NoiseGenerator(
        noise_type="homoscedastic", noise_stddev=0.1, rng_seed=RNG_SEED)
    eval_outputs = evaluation.evaluate(
        nasbench=self._nasbench,
        noise_generator=noise_generator,
        model_spec=self._model_spec,
        run_idx=0
    )
    valid_acc, test_acc, train_time = eval_outputs
    expected_noise_generator = noise_generator_lib.NoiseGenerator(
        noise_type="homoscedastic", noise_stddev=0.1, rng_seed=RNG_SEED)
    noise = expected_noise_generator.generate_noise()
    stats = computed_stats.COMPUTED_STATS[self._model_spec.graph_hash][108][0]
    self.assertEqual(valid_acc, stats["final_validation_accuracy"] + noise)
    self.assertEqual(test_acc, stats["final_test_accuracy"])
    self.assertEqual(train_time, stats["final_training_time"])

    # Not reproducible because generator bit_gen internal state has moved.
    eval_outputs = evaluation.evaluate(
        nasbench=self._nasbench,
        noise_generator=noise_generator,
        model_spec=self._model_spec,
        run_idx=0
    )
    valid_acc, test_acc, train_time = eval_outputs
    stats = computed_stats.COMPUTED_STATS[self._model_spec.graph_hash][108][0]
    self.assertNotEqual(valid_acc, stats["final_validation_accuracy"] + noise)
    self.assertEqual(test_acc, stats["final_test_accuracy"])
    self.assertEqual(train_time, stats["final_training_time"])
    # Reproducible with new generator using same seed.
    noise_generator = noise_generator_lib.NoiseGenerator(
        noise_type="homoscedastic", noise_stddev=0.1, rng_seed=RNG_SEED)
    eval_outputs = evaluation.evaluate(
        nasbench=self._nasbench,
        noise_generator=noise_generator,
        model_spec=self._model_spec,
        run_idx=0
    )
    valid_acc, test_acc, train_time = eval_outputs
    expected_noise_generator = noise_generator_lib.NoiseGenerator(
        noise_type="homoscedastic", noise_stddev=0.1, rng_seed=RNG_SEED)
    noise = expected_noise_generator.generate_noise()
    stats = computed_stats.COMPUTED_STATS[self._model_spec.graph_hash][108][0]
    self.assertEqual(valid_acc, stats["final_validation_accuracy"] + noise)
    self.assertEqual(test_acc, stats["final_test_accuracy"])
    self.assertEqual(train_time, stats["final_training_time"])

  def test_evaluate_no_noise_all_runs_test_acc_agg_mean(self):
    noise_generator = noise_generator_lib.NoiseGenerator(
        noise_type="", noise_stddev=0.0, rng_seed=RNG_SEED)
    stats = computed_stats.COMPUTED_STATS[self._model_spec.graph_hash][108]
    mean_test_acc = sum(
        [stats[i]["final_test_accuracy"] for i in range(3)]) / 3.0
    # Run 0.
    eval_outputs = evaluation.evaluate(
        nasbench=self._nasbench,
        noise_generator=noise_generator,
        model_spec=self._model_spec,
        run_idx=0,
        test_accuracy_aggregation_method="mean"
    )
    valid_acc, test_acc, train_time = eval_outputs
    self.assertEqual(valid_acc, stats[0]["final_validation_accuracy"])
    self.assertEqual(test_acc, mean_test_acc)
    self.assertEqual(train_time, stats[0]["final_training_time"])

    # Run 1.
    eval_outputs = evaluation.evaluate(
        nasbench=self._nasbench,
        noise_generator=noise_generator,
        model_spec=self._model_spec,
        run_idx=1,
        test_accuracy_aggregation_method="mean"
    )
    valid_acc, test_acc, train_time = eval_outputs
    self.assertEqual(valid_acc, stats[1]["final_validation_accuracy"])
    self.assertEqual(test_acc, mean_test_acc)
    self.assertEqual(train_time, stats[1]["final_training_time"])

    # Run 2.
    eval_outputs = evaluation.evaluate(
        nasbench=self._nasbench,
        noise_generator=noise_generator,
        model_spec=self._model_spec,
        run_idx=2,
        test_accuracy_aggregation_method="mean"
    )
    valid_acc, test_acc, train_time = eval_outputs
    self.assertEqual(valid_acc, stats[2]["final_validation_accuracy"])
    self.assertEqual(test_acc, mean_test_acc)
    self.assertEqual(train_time, stats[2]["final_training_time"])

  def test_evaluate_no_noise_all_runs_test_acc_agg_median(self):
    noise_generator = noise_generator_lib.NoiseGenerator(
        noise_type="", noise_stddev=0.0, rng_seed=RNG_SEED)
    stats = computed_stats.COMPUTED_STATS[self._model_spec.graph_hash][108]
    median_test_acc = sorted(
        [stats[i]["final_test_accuracy"] for i in range(3)])[1]
    # Run 0.
    eval_outputs = evaluation.evaluate(
        nasbench=self._nasbench,
        noise_generator=noise_generator,
        model_spec=self._model_spec,
        run_idx=0,
        test_accuracy_aggregation_method="median"
    )
    valid_acc, test_acc, train_time = eval_outputs
    self.assertEqual(valid_acc, stats[0]["final_validation_accuracy"])
    self.assertEqual(test_acc, median_test_acc)
    self.assertEqual(train_time, stats[0]["final_training_time"])

    # Run 1.
    eval_outputs = evaluation.evaluate(
        nasbench=self._nasbench,
        noise_generator=noise_generator,
        model_spec=self._model_spec,
        run_idx=1,
        test_accuracy_aggregation_method="median"
    )
    valid_acc, test_acc, train_time = eval_outputs
    self.assertEqual(valid_acc, stats[1]["final_validation_accuracy"])
    self.assertEqual(test_acc, median_test_acc)
    self.assertEqual(train_time, stats[1]["final_training_time"])

    # Run 2.
    eval_outputs = evaluation.evaluate(
        nasbench=self._nasbench,
        noise_generator=noise_generator,
        model_spec=self._model_spec,
        run_idx=2,
        test_accuracy_aggregation_method="median"
    )
    valid_acc, test_acc, train_time = eval_outputs
    self.assertEqual(valid_acc, stats[2]["final_validation_accuracy"])
    self.assertEqual(test_acc, median_test_acc)
    self.assertEqual(train_time, stats[2]["final_training_time"])

  def test_evaluate_no_noise_all_runs_test_acc_agg_min(self):
    noise_generator = noise_generator_lib.NoiseGenerator(
        noise_type="", noise_stddev=0.0, rng_seed=RNG_SEED)
    stats = computed_stats.COMPUTED_STATS[self._model_spec.graph_hash][108]
    min_test_acc = min([stats[i]["final_test_accuracy"] for i in range(3)])
    # Run 0.
    eval_outputs = evaluation.evaluate(
        nasbench=self._nasbench,
        noise_generator=noise_generator,
        model_spec=self._model_spec,
        run_idx=0,
        test_accuracy_aggregation_method="min"
    )
    valid_acc, test_acc, train_time = eval_outputs
    self.assertEqual(valid_acc, stats[0]["final_validation_accuracy"])
    self.assertEqual(test_acc, min_test_acc)
    self.assertEqual(train_time, stats[0]["final_training_time"])

    # Run 1.
    eval_outputs = evaluation.evaluate(
        nasbench=self._nasbench,
        noise_generator=noise_generator,
        model_spec=self._model_spec,
        run_idx=1,
        test_accuracy_aggregation_method="min"
    )
    valid_acc, test_acc, train_time = eval_outputs
    self.assertEqual(valid_acc, stats[1]["final_validation_accuracy"])
    self.assertEqual(test_acc, min_test_acc)
    self.assertEqual(train_time, stats[1]["final_training_time"])

    # Run 2.
    eval_outputs = evaluation.evaluate(
        nasbench=self._nasbench,
        noise_generator=noise_generator,
        model_spec=self._model_spec,
        run_idx=2,
        test_accuracy_aggregation_method="min"
    )
    valid_acc, test_acc, train_time = eval_outputs
    self.assertEqual(valid_acc, stats[2]["final_validation_accuracy"])
    self.assertEqual(test_acc, min_test_acc)
    self.assertEqual(train_time, stats[2]["final_training_time"])

  def test_evaluate_no_noise_all_runs_test_acc_agg_max(self):
    noise_generator = noise_generator_lib.NoiseGenerator(
        noise_type="", noise_stddev=0.0, rng_seed=RNG_SEED)
    stats = computed_stats.COMPUTED_STATS[self._model_spec.graph_hash][108]
    max_test_acc = max([stats[i]["final_test_accuracy"] for i in range(3)])
    # Run 0.
    eval_outputs = evaluation.evaluate(
        nasbench=self._nasbench,
        noise_generator=noise_generator,
        model_spec=self._model_spec,
        run_idx=0,
        test_accuracy_aggregation_method="max"
    )
    valid_acc, test_acc, train_time = eval_outputs
    self.assertEqual(valid_acc, stats[0]["final_validation_accuracy"])
    self.assertEqual(test_acc, max_test_acc)
    self.assertEqual(train_time, stats[0]["final_training_time"])

    # Run 1.
    eval_outputs = evaluation.evaluate(
        nasbench=self._nasbench,
        noise_generator=noise_generator,
        model_spec=self._model_spec,
        run_idx=1,
        test_accuracy_aggregation_method="max"
    )
    valid_acc, test_acc, train_time = eval_outputs
    self.assertEqual(valid_acc, stats[1]["final_validation_accuracy"])
    self.assertEqual(test_acc, max_test_acc)
    self.assertEqual(train_time, stats[1]["final_training_time"])

    # Run 2.
    eval_outputs = evaluation.evaluate(
        nasbench=self._nasbench,
        noise_generator=noise_generator,
        model_spec=self._model_spec,
        run_idx=2,
        test_accuracy_aggregation_method="max"
    )
    valid_acc, test_acc, train_time = eval_outputs
    self.assertEqual(valid_acc, stats[2]["final_validation_accuracy"])
    self.assertEqual(test_acc, max_test_acc)
    self.assertEqual(train_time, stats[2]["final_training_time"])


if __name__ == "__main__":
  absltest.main()
