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

"""Module for blocklisting imports."""

import sys
import nasbench
from nasbench import lib


class ImportBlocker(object):
  """Class for blocklisting imports."""

  def __init__(self, *args):
    self.module_names = args

  def find_module(self, fullname, path=None):
    _ = path
    if fullname in self.module_names:
      return self
    return None

  def create_module(self, _):
    return None

  def exec_module(self, _):
    # return an empty namespace
    return {}

# Blocklist nasbench.lib.evaluate as it depends on deprecated tensorflow 1.x
# and is unneeded here, but is imported by the necessary api module.
sys.modules['nasbench.lib.evaluate'] = ImportBlocker('nasbench.lib.evaluate')
