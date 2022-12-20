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

"""Tests for spec_validator_util."""

from unittest import mock

from absl.testing import absltest
from unified_functional_hashing.nasbench.python import spec_validator_util
import numpy as np


class SpecValidatorUtilTest(absltest.TestCase):

  def test_spec_valid(self):
    mock_model_spec = mock.MagicMock()
    mock_model_spec.matrix = np.array(
        [[0, 1, 1, 1, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0]]
    )
    mock_model_spec.ops = [
        "input",
        "conv1x1-bn-relu",
        "conv3x3-bn-relu",
        "conv3x3-bn-relu",
        "conv3x3-bn-relu",
        "maxpool3x3",
        "output"
    ]
    self.assertTrue(spec_validator_util.is_valid(mock_model_spec))

  def test_not_valid_spec(self):
    mock_model_spec = mock.MagicMock()
    mock_model_spec.matrix = np.array(
        [[0, 1, 1, 1, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0]]
    )
    mock_model_spec.ops = [
        "input",
        "conv1x1-bn-relu",
        "conv3x3-bn-relu",
        "conv3x3-bn-relu",
        "conv3x3-bn-relu",
        "maxpool3x3",
        "output"
    ]
    mock_model_spec.valid_spec = False  # valid attribute set to False
    self.assertFalse(spec_validator_util.is_valid(mock_model_spec))

  def test_too_many_vertices(self):
    mock_model_spec = mock.MagicMock()
    mock_model_spec.matrix = np.array(
        [[0, 1, 1, 1, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0]]
    )
    mock_model_spec.ops = [
        "input",
        "conv1x1-bn-relu",
        "conv3x3-bn-relu",
        "conv3x3-bn-relu",
        "conv3x3-bn-relu",
        "conv3x3-bn-relu",  # one too many
        "maxpool3x3",
        "output"
    ]
    self.assertFalse(spec_validator_util.is_valid(mock_model_spec))

  def test_too_many_edges(self):
    mock_model_spec = mock.MagicMock()
    mock_model_spec.matrix = np.array(
        [[0, 1, 1, 1, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 1]]  # extra 1 here
    )
    mock_model_spec.ops = [
        "input",
        "conv1x1-bn-relu",
        "conv3x3-bn-relu",
        "conv3x3-bn-relu",
        "conv3x3-bn-relu",
        "maxpool3x3",
        "output"
    ]
    self.assertFalse(spec_validator_util.is_valid(mock_model_spec))

  def test_too_no_input(self):
    mock_model_spec = mock.MagicMock()
    mock_model_spec.matrix = np.array(
        [[0, 1, 1, 1, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0]]
    )
    mock_model_spec.ops = [
        "conv1x1-bn-relu",  # no input
        "conv1x1-bn-relu",
        "conv3x3-bn-relu",
        "conv3x3-bn-relu",
        "conv3x3-bn-relu",
        "maxpool3x3",
        "output"
    ]
    self.assertFalse(spec_validator_util.is_valid(mock_model_spec))

  def test_too_no_output(self):
    mock_model_spec = mock.MagicMock()
    mock_model_spec.matrix = np.array(
        [[0, 1, 1, 1, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0]]
    )
    mock_model_spec.ops = [
        "input",
        "conv1x1-bn-relu",
        "conv3x3-bn-relu",
        "conv3x3-bn-relu",
        "conv3x3-bn-relu",
        "conv3x3-bn-relu",
        "maxpool3x3"  # no output
    ]
    self.assertFalse(spec_validator_util.is_valid(mock_model_spec))

  def test_not_available_op(self):
    mock_model_spec = mock.MagicMock()
    mock_model_spec.matrix = np.array(
        [[0, 1, 1, 1, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0]]
    )
    mock_model_spec.ops = [
        "input",
        "conv1x1-bn-relu",
        "conv3x3-bn-relu",
        "unknown_op",  # not available op
        "conv3x3-bn-relu",
        "maxpool3x3",
        "output"
    ]
    self.assertFalse(spec_validator_util.is_valid(mock_model_spec))


if __name__ == "__main__":
  absltest.main()
