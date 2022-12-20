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

"""Tests for noise_generator."""

from absl.testing import absltest
from unified_functional_hashing.nasbench.python import noise_generator as noise_generator_lib
import numpy as np


NOISE_DELTA = 0.01
NUM_SAMPLES = 1000000
RNG_SEED = 42


class NoiseGeneratorTest(absltest.TestCase):

  def test_no_noise_type(self):
    noise_generator = noise_generator_lib.NoiseGenerator(
        noise_type="", noise_stddev=0.0, rng_seed=RNG_SEED)
    noise_samples = [
        noise_generator.generate_noise() for _ in range(NUM_SAMPLES)]
    self.assertEqual(np.mean(noise_samples), 0.0)
    self.assertEqual(np.std(noise_samples), 0.0)
    self.assertEqual(np.sum(noise_samples), 0.0)

    noise_generator2 = noise_generator_lib.NoiseGenerator(
        noise_type="", noise_stddev=1.0, rng_seed=RNG_SEED)
    noise_samples = [
        noise_generator2.generate_noise() for _ in range(NUM_SAMPLES)]
    self.assertEqual(np.mean(noise_samples), 0.0)
    self.assertEqual(np.std(noise_samples), 0.0)
    self.assertEqual(np.sum(noise_samples), 0.0)

  def test_homoscedastic_no_noise(self):
    noise_generator = noise_generator_lib.NoiseGenerator(
        noise_type="homoscedastic", noise_stddev=0.0, rng_seed=RNG_SEED)
    noise_samples = [
        noise_generator.generate_noise() for _ in range(NUM_SAMPLES)]
    self.assertEqual(np.mean(noise_samples), 0.0)
    self.assertEqual(np.std(noise_samples), 0.0)
    self.assertEqual(np.sum(noise_samples), 0.0)

  def test_homoscedastic_noise(self):
    noise_generator = noise_generator_lib.NoiseGenerator(
        noise_type="homoscedastic", noise_stddev=1.0, rng_seed=RNG_SEED)
    noise_samples = [
        noise_generator.generate_noise() for _ in range(NUM_SAMPLES)]
    self.assertAlmostEqual(np.mean(noise_samples), 0.0, delta=NOISE_DELTA)
    self.assertAlmostEqual(np.std(noise_samples), 1.0, delta=NOISE_DELTA)

if __name__ == "__main__":
  absltest.main()
