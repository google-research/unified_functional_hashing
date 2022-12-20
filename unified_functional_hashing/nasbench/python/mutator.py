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

"""Mutators for NASBench.
"""

import copy
from typing import Optional

from unified_functional_hashing.nasbench.python import constants
from nasbench import api
import numpy as np


class Mutator():
  """Mutator that mutates ModelSpecs.
  """

  def __init__(self,
               nasbench: api.NASBench,
               mutation_rate: float = 1.0,
               rng_seed: Optional[int] = None):
    """Initializes Mutator.

    Mutation logic copied from Nasbench 101
    (https://github.com/google-research/nasbench/blob/master/NASBench.ipynb).

    Args:
      nasbench: NASBench instance.
      mutation_rate: Probability to mutate.
      rng_seed: Seed to initialize rng.
    """
    self.nasbench = nasbench
    self.mutation_rate = mutation_rate
    self.rng = np.random.default_rng(rng_seed)

  def mutate_spec(self, old_spec: api.ModelSpec) -> api.ModelSpec:
    """Computes a valid mutated spec from the old_spec.

    Args:
      old_spec: Parent ModelSpec.

    Returns:
      Mutated ModelSpec.
    """
    while True:
      new_matrix = copy.deepcopy(old_spec.original_matrix)
      new_ops = copy.deepcopy(old_spec.original_ops)

      # In expectation, V edges flipped (note that most end up being pruned).
      edge_mutation_prob = self.mutation_rate / constants.NUM_VERTICES
      for src in range(0, constants.NUM_VERTICES - 1):
        for dst in range(src + 1, constants.NUM_VERTICES):
          if self.rng.random(size=1)[0] < edge_mutation_prob:
            new_matrix[src, dst] = 1 - new_matrix[src, dst]

      # In expectation, one op is resampled.
      op_mutation_prob = self.mutation_rate / constants.OP_SPOTS
      for ind in range(1, constants.NUM_VERTICES - 1):
        if self.rng.random(size=1)[0] < op_mutation_prob:
          available = [
              o for o in self.nasbench.config["nasbench_available_ops"]
              if o != new_ops[ind]
          ]
          new_ops[ind] = self.rng.choice(available)

      new_spec = api.ModelSpec(new_matrix, new_ops)
      if self.nasbench.is_valid(new_spec):
        return new_spec
