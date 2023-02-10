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

"""Noise generator for NASBench.
"""

from typing import Optional

import numpy as np


class NoiseGenerator():
  """Noise generator of various types.
  """

  def __init__(self,
               noise_type: str,
               noise_stddev: float,
               rng_seed: Optional[int] = None):
    """Initializes NoiseGenerator.

    Args:
      noise_type: type of noise to add to the evaluated validation accuracy.
      noise_stddev: noise standard deviation for evaluated validation accuracy.
      rng_seed: Seed to initialize rng.
    """
    self.rng = np.random.default_rng(rng_seed)
    self.noise_type = noise_type
    self.noise_stddev = noise_stddev

  def generate_noise(self) -> float:
    """Evaluates model spec to get accuracies.

    Returns:
      Float noise.
    """
    if self.noise_type == "homoscedastic":
      return self.rng.normal(loc=0.0, scale=self.noise_stddev)
    return 0.0
