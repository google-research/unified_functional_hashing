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

"""Classes for storing experiment statistics and metadata.
"""


class ExperimentMetadataRepeat:
  """Metadata of potential interest for a single repeat of an experiment."""

  def __init__(self, repeat_idx):
    """Initializes ExperimentMetadataRepeat.

    Args:
      repeat_idx: The index of the repeat within its experiment.
    """
    self.repeat_idx = repeat_idx
    self.times = []
    self.best_valids = []
    self.best_tests = []
    self.num_cache_misses = []
    self.num_cache_partial_hits = []
    self.num_cache_full_hits = []
    self.cache_size = []
    self.model_hashes = []
    self.run_indices = []
    self.validation_accuracies = []
    self.test_accuracies = []
    self.training_times = []
    self.graph_hashes = []


class ExperimentMetadata:
  """Metadata of potential interest for a all repeats of an experiment."""

  def __init__(self, experiment_id):
    """Initializes ExperimentMetadata.

    Args:
      experiment_id: The id given to the experiment.
    """
    self.experiment_id = experiment_id
    self.repeat_metadata = []
