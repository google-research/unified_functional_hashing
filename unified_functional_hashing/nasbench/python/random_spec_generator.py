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

"""Random spec generator for NASBench.
"""

from typing import Optional

from unified_functional_hashing.nasbench.python import constants
from nasbench import api
import numpy as np


class RandomSpecGenerator():
  """RandomSpecGenerator that generates random ModelSpecs.
  """

  def __init__(self,
               nasbench: api.NASBench,
               rng_seed: Optional[int] = None):
    """Initializes RandomSpecGenerator.

    Args:
      nasbench: NASBench instance.
      rng_seed: Seed to initialize rng.
    """
    self.nasbench = nasbench
    self.rng = np.random.default_rng(rng_seed)

  def random_spec(self) -> api.ModelSpec:
    """Returns a random valid spec.

    Returns:
      Random valid ModelSpec.
    """
    while True:
      matrix = self.rng.choice(
          constants.ALLOWED_EDGES,
          size=(constants.NUM_VERTICES, constants.NUM_VERTICES))
      matrix = np.triu(matrix, 1)
      ops = self.rng.choice(
          constants.ALLOWED_OPS, size=(constants.NUM_VERTICES)).tolist()
      ops[0] = constants.INPUT
      ops[-1] = constants.OUTPUT
      spec = api.ModelSpec(matrix=matrix, ops=ops)
      if self.nasbench.is_valid(spec):
        return spec
