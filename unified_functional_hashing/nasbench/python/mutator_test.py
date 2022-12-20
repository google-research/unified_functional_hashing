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

"""Tests for mutator."""

import copy
from typing import Tuple
from unittest import mock

from absl.testing import absltest
from unified_functional_hashing.nasbench.python import constants
from unified_functional_hashing.nasbench.python import mutator as mutator_lib
from unified_functional_hashing.nasbench.python import spec_validator_util
from unified_functional_hashing.nasbench.python import testing
from nasbench import api
import numpy as np


RNG_SEED = 42


def get_mutated_spec_matrix_and_ops(
    mutator: mutator_lib.Mutator,
    parent_spec: api.ModelSpec) -> Tuple[str, str]:
  """Gets mutated ModelSpec matrix and ops.

  Args:
    mutator: Mutator instance.
    parent_spec: Parent ModelSpec.

  Returns:
    Child ModelSpec matrix string and ops string.
  """
  child_spec = mutator.mutate_spec(old_spec=parent_spec)
  matrix_str_list = [str(x) for x in child_spec.matrix.flatten().tolist()]
  ops_str_list = child_spec.ops
  return ",".join(matrix_str_list), ",".join(ops_str_list)


class MutatorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    mock_nasbench = mock.MagicMock()
    mock_nasbench.config = constants.NASBENCH_CONFIG
    mock_nasbench.is_valid = mock.MagicMock(
        side_effect=spec_validator_util.is_valid)
    self._nasbench = mock_nasbench

    mock_model_spec = mock.MagicMock()
    mock_model_spec.original_matrix = np.array(
        [[0, 1, 1, 1, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0]]
    )
    mock_model_spec.original_ops = [
        "input",
        "conv1x1-bn-relu",
        "conv3x3-bn-relu",
        "conv3x3-bn-relu",
        "conv3x3-bn-relu",
        "maxpool3x3",
        "output"
    ]
    mock_model_spec.matrix = copy.deepcopy(mock_model_spec.original_matrix)
    mock_model_spec.ops = copy.deepcopy(mock_model_spec.original_ops)
    mock_model_spec.data_format = "channels_last"
    self._model_spec = mock_model_spec

  def test_doesnt_mutate_original_model_spec_object(self):
    mutator = mutator_lib.Mutator(
        nasbench=self._nasbench, mutation_rate=1.0, rng_seed=RNG_SEED)
    _ = mutator.mutate_spec(old_spec=self._model_spec)
    expected_matrix = np.array(
        [[0, 1, 1, 1, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0]]
    )
    expected_ops = [
        "input",
        "conv1x1-bn-relu",
        "conv3x3-bn-relu",
        "conv3x3-bn-relu",
        "conv3x3-bn-relu",
        "maxpool3x3",
        "output"
    ]
    self.assertListEqual(
        expected_matrix.flatten().tolist(),
        self._model_spec.original_matrix.flatten().tolist())
    self.assertListEqual(expected_ops, self._model_spec.original_ops)
    self.assertListEqual(
        expected_matrix.flatten().tolist(),
        self._model_spec.matrix.flatten().tolist())
    self.assertListEqual(expected_ops, self._model_spec.ops)

  def test_mutates(self):
    mutator = mutator_lib.Mutator(
        nasbench=self._nasbench, mutation_rate=1.0, rng_seed=RNG_SEED)
    new_spec = mutator.mutate_spec(old_spec=self._model_spec)
    expected_matrix = np.array(
        [[0, 1, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 1],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 1, 0, 1],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0]]
    )
    expected_ops = [
        "input",
        "conv1x1-bn-relu",
        "conv3x3-bn-relu",
        "conv3x3-bn-relu",
        "conv3x3-bn-relu",
        "output"
    ]
    self.assertListEqual(
        expected_matrix.flatten().tolist(), expected_matrix.flatten().tolist())
    self.assertListEqual(expected_ops, new_spec.ops)

  def test_doesnt_mutate(self):
    mutator = mutator_lib.Mutator(
        nasbench=self._nasbench, mutation_rate=0.0, rng_seed=RNG_SEED)
    new_spec = mutator.mutate_spec(old_spec=self._model_spec)
    expected_matrix = np.array(
        [[0, 1, 1, 1, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0]]
    )
    expected_ops = [
        "input",
        "conv1x1-bn-relu",
        "conv3x3-bn-relu",
        "conv3x3-bn-relu",
        "conv3x3-bn-relu",
        "maxpool3x3",
        "output"
    ]
    self.assertListEqual(
        expected_matrix.flatten().tolist(), expected_matrix.flatten().tolist())
    self.assertListEqual(expected_ops, new_spec.ops)

  def test_spec_eventually(self):
    mutator = mutator_lib.Mutator(
        nasbench=self._nasbench, mutation_rate=1.0, rng_seed=RNG_SEED)

    expected_matrix = np.array(
        [[0, 1, 1, 0, 1, 0],
         [0, 0, 0, 0, 0, 1],
         [0, 0, 0, 1, 1, 0],
         [0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0]])

    expected_ops = [
        "input",
        "conv3x3-bn-relu",
        "conv3x3-bn-relu",
        "conv3x3-bn-relu",
        "maxpool3x3",
        "output"
    ]

    self.assertTrue(
        testing.is_eventually(
            func=lambda:  # pylint: disable=g-long-lambda
            get_mutated_spec_matrix_and_ops(
                mutator=mutator,
                parent_spec=self._model_spec),
            required_values=[
                (",".join([str(x) for x in expected_matrix.flatten().tolist()]),
                 ",".join(expected_ops))],
            allowed_values=None,
            max_run_time=30.0))

if __name__ == "__main__":
  absltest.main()
