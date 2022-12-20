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

"""Convenience utilities for testing methods that return random values."""

import time
from typing import Callable, List, Optional, TypeVar


SomeType = TypeVar("SomeType")  # pylint: disable=invalid-name


def is_eventually(
    func: Callable[[], SomeType],
    required_values: List[SomeType],
    allowed_values: Optional[List[SomeType]],
    max_run_time: float = 3.0) -> bool:
  """Checks that repeatedly calling a function produces the desired values.

  Args:
    func: a function to be called repeatedly. It need not always return the same
      value.
    required_values: func will be retried until all values in this list are
      returned. Must be non-empty.
    allowed_values: the assertion will fail if func ever returns a value not in
      this list. `None` means all values are allowed.
    max_run_time: the assertion will fail if more time than this is needed to
      generate all the required values. In seconds. `None` means no time limit.

  Returns:
    Whether successful.

  Raises:
    RandomTestingError: if the assertion fails.
  """
  assert required_values
  seen_values_set = set()
  required_values_set = set(required_values)
  start_time = time.time()
  while max_run_time is None or time.time() - start_time < max_run_time:
    if seen_values_set == required_values_set:
      return True  # Assertion succeeded.

    value = func()
    if allowed_values is not None and value not in allowed_values:
      print("Generated unexpected value: %s." % str(value))
      return False
    if value in required_values_set:
      seen_values_set.add(value)

  missing_values = [v for v in required_values if v not in seen_values_set]
  print(
      "Took too long to generate all required values. Missing values: %s." %
      str([str(v) for v in missing_values]))
  return False
