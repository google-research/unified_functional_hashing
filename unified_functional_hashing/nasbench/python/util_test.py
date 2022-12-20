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

"""Tests for util."""

from absl.testing import absltest
from unified_functional_hashing.nasbench.python import util


class UtilTest(absltest.TestCase):

  def test_best_lists_grow(self):
    fitnesses = (0.2, 0.1)
    best_valids = [0.0]
    best_tests = [0.0]
    util.update_best_fitnesses(fitnesses, best_valids, best_tests)
    self.assertLen(best_valids, 2)
    self.assertLen(best_tests, 2)

  def test_better_valid_better_test_fitness(self):
    fitnesses = (0.4, 0.3)
    best_valids = [0.0, 0.2]
    best_tests = [0.0, 0.1]
    util.update_best_fitnesses(fitnesses, best_valids, best_tests)
    self.assertListEqual(best_valids, [0.0, 0.2, 0.4])
    self.assertListEqual(best_tests, [0.0, 0.1, 0.3])

  def test_better_valid_worse_test_fitness(self):
    fitnesses = (0.4, 0.05)
    best_valids = [0.0, 0.2]
    best_tests = [0.0, 0.1]
    util.update_best_fitnesses(fitnesses, best_valids, best_tests)
    self.assertListEqual(best_valids, [0.0, 0.2, 0.4])
    self.assertListEqual(best_tests, [0.0, 0.1, 0.05])

  def test_worse_valid_better_test_fitness(self):
    fitnesses = (0.15, 0.6)
    best_valids = [0.0, 0.2]
    best_tests = [0.0, 0.1]
    util.update_best_fitnesses(fitnesses, best_valids, best_tests)
    self.assertListEqual(best_valids, [0.0, 0.2, 0.2])
    self.assertListEqual(best_tests, [0.0, 0.1, 0.1])

  def test_worse_valid_worse_test_fitness(self):
    fitnesses = (0.15, 0.05)
    best_valids = [0.0, 0.2]
    best_tests = [0.0, 0.1]
    util.update_best_fitnesses(fitnesses, best_valids, best_tests)
    self.assertListEqual(best_valids, [0.0, 0.2, 0.2])
    self.assertListEqual(best_tests, [0.0, 0.1, 0.1])


if __name__ == '__main__':
  absltest.main()
