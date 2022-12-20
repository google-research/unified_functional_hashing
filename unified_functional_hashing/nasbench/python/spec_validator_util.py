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

"""Spec validator utility for NASBench.
"""

from unified_functional_hashing.nasbench.python import constants
import numpy as np


class OutOfDomainError(Exception):
  """Indicates that the requested graph is outside of the search domain."""


def is_valid(model_spec):
  """Checks the validity of the model_spec.

  For the purposes of benchmarking, this does not increment the budget
  counters.

  Args:
    model_spec: ModelSpec object.

  Returns:
    True if model is within space.
  """
  try:
    _check_spec(model_spec)
  except OutOfDomainError:
    return False

  return True


def _check_spec(model_spec):
  """Checks that the model spec is within the dataset."""
  if not model_spec.valid_spec:
    raise OutOfDomainError("invalid spec, provided graph is disconnected.")

  num_vertices = len(model_spec.ops)
  num_edges = np.sum(model_spec.matrix)

  if num_vertices > constants.NASBENCH_CONFIG["nasbench_module_vertices"]:
    raise OutOfDomainError(
        "too many vertices, got %d (max vertices = %d)"
        % (num_vertices, constants.NASBENCH_CONFIG["nasbench_module_vertices"]))

  if num_edges > constants.NASBENCH_CONFIG["nasbench_max_edges"]:
    raise OutOfDomainError(
        "too many edges, got %d (max edges = %d)"
        % (num_edges, constants.NASBENCH_CONFIG["nasbench_max_edges"]))

  if model_spec.ops[0] != "input":
    raise OutOfDomainError("first operation should be \"input\"")
  if model_spec.ops[-1] != "output":
    raise OutOfDomainError("last operation should be \"output\"")
  for op in model_spec.ops[1:-1]:
    if op not in constants.NASBENCH_CONFIG["nasbench_available_ops"]:
      raise OutOfDomainError(
          "unsupported op %s (available ops = %s)"
          % (op, constants.NASBENCH_CONFIG["nasbench_available_ops"]))
