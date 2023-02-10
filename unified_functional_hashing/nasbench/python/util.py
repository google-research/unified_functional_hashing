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

"""Common NASBench utilities.
"""

from typing import List, Tuple


def update_best_fitnesses(fitnesses: Tuple[float, float],
                          best_valids: List[float],
                          best_tests: List[float]) -> None:
  """Updates best fitnesses.

  Args:
    fitnesses: Tuple of valid and test fitness.
    best_valids: List of best valid fitnesses.
    best_tests: List of best test fitnesses.
  """
  if fitnesses[0] > best_valids[-1]:
    best_valids.append(fitnesses[0])
    best_tests.append(fitnesses[1])
  else:
    best_valids.append(best_valids[-1])
    best_tests.append(best_tests[-1])
