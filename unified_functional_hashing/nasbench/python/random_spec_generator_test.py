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

"""Tests for random_spec_generator."""

from typing import Tuple
from unittest import mock

from absl.testing import absltest
from unified_functional_hashing.nasbench.python import constants
from unified_functional_hashing.nasbench.python import random_spec_generator as random_spec_generator_lib
from unified_functional_hashing.nasbench.python import spec_validator_util
from unified_functional_hashing.nasbench.python import testing
import numpy as np


RNG_SEED = 42


def get_random_spec_matrix_and_ops(
    random_spec_generator: random_spec_generator_lib.RandomSpecGenerator
    ) -> Tuple[str, str]:
  """Gets random ModelSpec matrix and ops.

  Args:
    random_spec_generator: RandomSpecGenerator instance.

  Returns:
    Random ModelSpec matrix string and ops string.
  """
  random_spec = random_spec_generator.random_spec()
  matrix_str_list = [str(x) for x in random_spec.matrix.flatten().tolist()]
  ops_str_list = random_spec.ops
  return ",".join(matrix_str_list), ",".join(ops_str_list)


class RandomSpecGeneratorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    mock_nasbench = mock.MagicMock()
    mock_nasbench.config = constants.NASBENCH_CONFIG
    mock_nasbench.is_valid = mock.MagicMock(
        side_effect=spec_validator_util.is_valid)
    self._nasbench = mock_nasbench

  def test_generates_random_spec(self):
    random_spec_generator = random_spec_generator_lib.RandomSpecGenerator(
        nasbench=self._nasbench, rng_seed=RNG_SEED)
    random_spec = random_spec_generator.random_spec()
    expected_matrix = np.array(
        [[0, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0]]
    )
    expected_ops = [
        "input",
        "conv1x1-bn-relu",
        "conv1x1-bn-relu",
        "maxpool3x3",
        "conv3x3-bn-relu",
        "output"
    ]
    self.assertListEqual(random_spec.matrix.flatten().tolist(),
                         expected_matrix.flatten().tolist())
    self.assertListEqual(random_spec.ops, expected_ops)

  def test_different_seeds_different_specs(self):
    random_spec_generator1 = random_spec_generator_lib.RandomSpecGenerator(
        nasbench=self._nasbench, rng_seed=RNG_SEED)
    random_spec1 = random_spec_generator1.random_spec()
    random_spec_generator2 = random_spec_generator_lib.RandomSpecGenerator(
        nasbench=self._nasbench, rng_seed=7)
    random_spec2 = random_spec_generator2.random_spec()
    self.assertFalse(
        np.all(random_spec1.original_matrix == random_spec2.original_matrix))
    self.assertFalse(
        np.all(
            np.array(random_spec1.original_ops) == np.array(
                random_spec2.original_ops)))

  def test_spec_matrix_has_correct_shape(self):
    random_spec_generator = random_spec_generator_lib.RandomSpecGenerator(
        nasbench=self._nasbench, rng_seed=RNG_SEED)
    random_spec = random_spec_generator.random_spec()
    expected_shape = (constants.NUM_VERTICES, constants.NUM_VERTICES)
    self.assertTupleEqual(random_spec.original_matrix.shape, expected_shape)
    # In case of pruning.
    self.assertLessEqual(random_spec.matrix.shape[0], expected_shape[0])
    self.assertLessEqual(random_spec.matrix.shape[1], expected_shape[1])

  def test_spec_matrix_has_correct_allowed_edges(self):
    random_spec_generator = random_spec_generator_lib.RandomSpecGenerator(
        nasbench=self._nasbench, rng_seed=RNG_SEED)
    random_spec = random_spec_generator.random_spec()
    self.assertTrue(
        np.all(np.isin(random_spec.original_matrix, constants.ALLOWED_EDGES)))
    # In case of pruning.
    self.assertTrue(
        np.all(np.isin(random_spec.matrix, constants.ALLOWED_EDGES)))

  def test_spec_ops_has_correct_len(self):
    random_spec_generator = random_spec_generator_lib.RandomSpecGenerator(
        nasbench=self._nasbench, rng_seed=RNG_SEED)
    random_spec = random_spec_generator.random_spec()
    self.assertLen(random_spec.original_ops, constants.NUM_VERTICES)
    # In case of pruning.
    self.assertLessEqual(len(random_spec.ops), constants.NUM_VERTICES)

  def test_spec_ops_have_correct_allowed_ops(self):
    random_spec_generator = random_spec_generator_lib.RandomSpecGenerator(
        nasbench=self._nasbench, rng_seed=RNG_SEED)
    random_spec = random_spec_generator.random_spec()
    self.assertTrue(
        np.all(
            np.isin(
                np.array(random_spec.original_ops[1:-1]),
                constants.ALLOWED_OPS)))
    # In case of pruning.
    self.assertTrue(
        np.all(
            np.isin(np.array(random_spec.ops[1:-1]), constants.ALLOWED_OPS)))

  def test_spec_ops_have_correct_input_and_output_ops(self):
    random_spec_generator = random_spec_generator_lib.RandomSpecGenerator(
        nasbench=self._nasbench, rng_seed=RNG_SEED)
    random_spec = random_spec_generator.random_spec()
    self.assertEqual(random_spec.original_ops[0], constants.INPUT)
    self.assertEqual(random_spec.original_ops[-1], constants.OUTPUT)
    # In case of pruning.
    self.assertEqual(random_spec.ops[0], constants.INPUT)
    self.assertEqual(random_spec.ops[-1], constants.OUTPUT)

  def test_spec_eventually(self):
    random_spec_generator = random_spec_generator_lib.RandomSpecGenerator(
        nasbench=self._nasbench, rng_seed=RNG_SEED)
    expected_matrix = np.array(
        [[0, 1, 0, 1, 1, 0],
         [0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 1],
         [0, 0, 0, 0, 1, 1],
         [0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0]])
    expected_ops = [
        "input",
        "maxpool3x3",
        "maxpool3x3",
        "maxpool3x3",
        "conv3x3-bn-relu",
        "output"
    ]
    self.assertTrue(
        testing.is_eventually(
            func=lambda: get_random_spec_matrix_and_ops(random_spec_generator),
            required_values=[
                (",".join([str(x) for x in expected_matrix.flatten().tolist()]),
                 ",".join(expected_ops))],
            allowed_values=None,
            max_run_time=30.0))


if __name__ == "__main__":
  absltest.main()
