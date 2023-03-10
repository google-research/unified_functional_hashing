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

"""Install requirements for unified_functional_hashing."""

from setuptools import find_packages
from setuptools import setup

setup(
    name='unified_functional_hashing',
    version='0.1',
    description='functional equivalence cache demonstration on NASBench101',
    author='Google LLC',
    author_email='noreply@google.com',
    url='https://github.com/google-research/unified_functional_hashing',
    packages=find_packages(),
    install_requires=[
        'absl-py',
        'numpy',
        'pytest',
        'tensorflow>=2.0.0',
        'nasbench @ git+https://github.com/google-research/nasbench@master',
    ],
    python_requires='>=3.6',
)
