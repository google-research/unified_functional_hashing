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

"""NASBench experiment worker code.
"""

from absl import app
from absl import flags
from unified_functional_hashing.nasbench.python import run_worker_util
from nasbench import api


# All these flags are expected by the launcher.
_EXPERIMENT_ID = flags.DEFINE_string(
    "experiment_id", "",
    "The ID of the experiment to run. Required.")
_EXPERIMENT_REPEAT_INDICES = flags.DEFINE_string(
    "experiment_repeat_indices", "0",
    "Comma-separated string of experiment repeat indices.")
_FEC_REMOVE_PROBABILITY = flags.DEFINE_float(
    "fec_remove_probability", 0.0,
    "After a cache hit, the probability a key will be removed from the cache.")
_FIXED_RUN_IDX = flags.DEFINE_integer(
    "fixed_run_idx", -1,
    "The run index to use for getting computed_stats. If -1, then will be "
    "random and unfixed.")
_HASHING_TIME = flags.DEFINE_float(
    "hashing_time", 10.0,
    "Number of seconds it takes to generate hash.")
_MANTISSA_BITS = flags.DEFINE_integer(
    "mantissa_bits", 24,
    "The number of bits to use in the mantissa.")
_MAX_NUM_EVALS = flags.DEFINE_integer(
    "max_num_evals", 1,
    "The maximum number of evaluations to perform for a hash.")
_MAX_TIME_BUDGET = flags.DEFINE_float(
    "max_time_budget", 5e6,
    "The maximum time budget for NASBench.")
_MUTATION_RATE = flags.DEFINE_float(
    "mutation_rate", 1.0,
    "The probability to mutate a spec.")
_NOISE_STDDEV = flags.DEFINE_float(
    "noise_stddev", 1.0,
    "The noise standard deviation for evaluated validation accuracy.")
_NOISE_TYPE = flags.DEFINE_string(
    "noise_type", "",
    "The type of noise to add to the evaluated validation accuracy.")
_POPULATION_SIZE = flags.DEFINE_integer(
    "population_size", 50,
    "The number of individuals in the population.")
_RNG_SEED = flags.DEFINE_integer(
    "rng_seed", -1,
    "Seed to initialize rng. If -1, then will be unseeded.")
_SAVE_CHILD_HISTORY = flags.DEFINE_bool(
    "save_child_history", False,
    "Whether to save history of child. If True, its isomorphism-invariant "
    "graph hash, run indices, validation & test accuracies, training times "
    "will be saved.")
_SAVE_FEC_HISTORY = flags.DEFINE_bool(
    "save_fec_history", False,
    "Whether to save FEC's history. If True, the FEC's number of cache misses, "
    "cache partial hits, cache full hits, cache size, and the Hasher's 4 "
    "accuracy model hash will be saved.")
_SEARCH_METHOD = flags.DEFINE_string(
    "search_method", "",
    "The search method to use. Required.")
_TFRECORD_CNS_PATH = flags.DEFINE_string(
    "tfrecord_cns_path", "",
    "CNS path to TFRecord."
    "Required.")
_TOURNAMENT_SIZE = flags.DEFINE_integer(
    "tournament_size", 10,
    "The number of individuals selected for a tournament.")
_USE_FEC = flags.DEFINE_bool(
    "use_fec", False,
    "Whether to use FEC or not.")
_MASTER = flags.DEFINE_string("master", "", "")


def run():
  """Runs experiment."""
  nasbench = api.NASBench(_TFRECORD_CNS_PATH.value)
  run_worker_util.run_experiment(
      nasbench=nasbench,
      experiment_repeat_indices=_EXPERIMENT_REPEAT_INDICES.value,
      fec_remove_probability=_FEC_REMOVE_PROBABILITY.value,
      fixed_run_idx=_FIXED_RUN_IDX.value,
      hashing_time=_HASHING_TIME.value,
      mantissa_bits=_MANTISSA_BITS.value,
      max_num_evals=_MAX_NUM_EVALS.value,
      max_time_budget=_MAX_TIME_BUDGET.value,
      mutation_rate=_MUTATION_RATE.value,
      noise_type=_NOISE_TYPE.value,
      noise_stddev=_NOISE_STDDEV.value,
      population_size=_POPULATION_SIZE.value,
      save_child_history=_SAVE_CHILD_HISTORY.value,
      save_fec_history=_SAVE_FEC_HISTORY.value,
      search_method=_SEARCH_METHOD.value,
      tournament_size=_TOURNAMENT_SIZE.value,
      use_fec=_USE_FEC.value,
      rng_seed=_RNG_SEED.value if _RNG_SEED.value > 0 else None
  )


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  run()


if __name__ == "__main__":
  app.run(main)
