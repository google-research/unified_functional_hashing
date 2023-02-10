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

"""Evaluation for NASBench.
"""

from typing import Tuple

from unified_functional_hashing.nasbench.python import noise_generator as noise_generator_lib
from nasbench import api


def evaluate(nasbench: api.NASBench,
             noise_generator: noise_generator_lib.NoiseGenerator,
             model_spec: api.ModelSpec,
             run_idx: int,
             test_accuracy_aggregation_method: str = "",
             test: bool = False) -> Tuple[float, float, float]:
  """Evaluates model spec to get eval outputs.

  Args:
    nasbench: NASBenech instance.
    noise_generator: NoiseGenerator instance.
    model_spec: ModelSpec matrix.
    run_idx: Index of run to extract stats from.
    test_accuracy_aggregation_method: The method to aggregate test accuracies.
        '', 'mean', 'median', 'min', 'max'.
    test: Whether called from a test or not.

  Returns:
    3-tuple of floats for valid and test accuracies and training time.
  """
  if test and not hasattr(model_spec, "graph_hash"):
    model_spec.graph_hash = model_spec.hash_spec(
        canonical_ops=nasbench.config["available_ops"])
  _, computed_stats = nasbench.get_metrics_from_spec(model_spec)
  valid_acc = computed_stats[108][run_idx]["final_validation_accuracy"]
  valid_acc += noise_generator.generate_noise()
  if not test_accuracy_aggregation_method:
    test_acc = computed_stats[108][run_idx]["final_test_accuracy"]
  else:
    test_accuracies = [computed_stats[108][i]["final_test_accuracy"]
                       for i in range(3)]
    if test_accuracy_aggregation_method == "mean":
      test_acc = sum(test_accuracies) / 3.0
    elif test_accuracy_aggregation_method == "median":
      test_acc = sorted(test_accuracies)[1]
    elif test_accuracy_aggregation_method == "min":
      test_acc = min(test_accuracies)
    elif test_accuracy_aggregation_method == "max":
      test_acc = max(test_accuracies)
    else:
      raise ValueError(
          "{} is not a valid test_accuracy_aggregation_method!".format(
              test_accuracy_aggregation_method))
  time = float(computed_stats[108][run_idx]["final_training_time"])
  return valid_acc, test_acc, time
