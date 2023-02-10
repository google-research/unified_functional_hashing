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

"""NASBench run_worker utilities."""

from typing import Optional, Union

from unified_functional_hashing.nasbench.python import constants
from unified_functional_hashing.nasbench.python import evaluator as evaluator_lib
from unified_functional_hashing.nasbench.python import evolution_search
from unified_functional_hashing.nasbench.python import experiment_data
from unified_functional_hashing.nasbench.python import random_search
from nasbench import api
import numpy as np

EXPERIMENT_METADATA_ID_LENGTH = 64


def build_searcher(
    nasbench: api.NASBench,
    evaluator: evaluator_lib.Evaluator,
    rng: np.random.default_rng,
    max_time_budget: float = 5e6,
    mutation_rate: float = 1.0,
    population_size: int = 50,
    save_child_history: bool = False,
    search_method: str = 'regularized_evolution',
    tournament_size: int = 10,
) -> Union[random_search.RandomSearcher, evolution_search.EvolutionSearcher]:
  """Builds searcher for experiment.

  Args:
      nasbench: NASBench instance.
      evaluator: Evaluator to calculate valid and test accuracies.
      rng: Numpy RNG from caller.
      max_time_budget: Maximum time budget for search.
      mutation_rate: The probability to mutate a ModelSpec.
      population_size: The number of individuals in the population.
      save_child_history: Whether to save history of child. If True, its
        isomorphism-invariant graph hash, run indices, validation & test
        accuracies, and training times will be saved.
      search_method: Method of search: random, regularized_evolution, or
        elitism_evolution.
      tournament_size: The number of individuals selected for a tournament.

  Returns:
    Instance of RandomSearcher or EvolutionSearcher
  """
  if search_method == 'random':
    searcher = random_search.RandomSearcher(
        nasbench=nasbench,
        evaluator=evaluator,
        max_time_budget=max_time_budget,
        save_child_history=save_child_history,
        rng_seed=rng.integers(low=1, high=constants.MAX_RNG_SEED, size=1)[0])
  elif (search_method == 'regularized_evolution' or
        search_method == 'elitism_evolution'):

    searcher_cls = evolution_search.EvolutionSearcher

    searcher = searcher_cls(
        nasbench=nasbench,
        evaluator=evaluator,
        evolution_type=search_method,
        max_time_budget=max_time_budget,
        population_size=population_size,
        save_child_history=save_child_history,
        tournament_size=tournament_size,
        mutation_rate=mutation_rate,
        rng_seed=rng.integers(low=1, high=constants.MAX_RNG_SEED, size=1)[0])
  else:
    raise ValueError('{} is not a valid search_method!'.format(search_method))
  return searcher


def create_experiment_repeat_metadata_records(
    repeat_idx: int, metadata_iterable_dict: dict[str, list[Union[float, int,
                                                                  str]]]
) -> experiment_data.ExperimentMetadataRepeat:
  """Creates `ExperimentMetadata` record.

  Args:
    repeat_idx: Experiment repeat index.
    metadata_iterable_dict: Dict of string keys and value float or int lists.

  Returns:
    `ExperimentMetadata` record.
  """
  experiment_metadata_id = str(repeat_idx)
  experiment_repeat_data = experiment_data.ExperimentMetadataRepeat(
      experiment_metadata_id)
  experiment_repeat_data.times = metadata_iterable_dict['times']
  experiment_repeat_data.best_valids = metadata_iterable_dict['best_valids']
  experiment_repeat_data.best_tests = metadata_iterable_dict['best_tests']
  if 'validation_accuracies' in metadata_iterable_dict:
    experiment_repeat_data.validation_accuracies = metadata_iterable_dict[
        'validation_accuracies']
  if 'test_accuracies' in metadata_iterable_dict:
    experiment_repeat_data.test_accuracies = metadata_iterable_dict[
        'test_accuracies']
  if 'training_times' in metadata_iterable_dict:
    experiment_repeat_data.training_times = metadata_iterable_dict[
        'training_times']
  if 'num_cache_misses' in metadata_iterable_dict:
    experiment_repeat_data.num_cache_misses = metadata_iterable_dict[
        'num_cache_misses']
  if 'num_cache_partial_hits' in metadata_iterable_dict:
    experiment_repeat_data.num_cache_partial_hits = metadata_iterable_dict[
        'num_cache_partial_hits']
  if 'num_cache_full_hits' in metadata_iterable_dict:
    experiment_repeat_data.num_cache_full_hits = metadata_iterable_dict[
        'num_cache_full_hits']
  if 'cache_size' in metadata_iterable_dict:
    experiment_repeat_data.cache_size = metadata_iterable_dict['cache_size']
  if 'model_hashes' in metadata_iterable_dict:
    experiment_repeat_data.model_hashes = metadata_iterable_dict['model_hashes']
  if 'run_indices' in metadata_iterable_dict:
    experiment_repeat_data.run_indices = metadata_iterable_dict['run_indices']
  if 'graph_hashes' in metadata_iterable_dict:
    experiment_repeat_data.graph_hashes = metadata_iterable_dict['graph_hashes']

  return experiment_repeat_data


def run_experiment_repeat(
    nasbench: api.NASBench,
    evaluator: evaluator_lib.Evaluator,
    rng: np.random.default_rng,
    experiment_metadata: experiment_data.ExperimentMetadata,
    max_time_budget: float = 5e6,
    mutation_rate: float = 1.0,
    population_size: int = 50,
    save_child_history: bool = False,
    search_method: str = 'regularized_evolution',
    tournament_size: int = 10,
    repeat_idx: int = 0) -> None:
  """Runs experiment repeat.

  Args:
      nasbench: NASBench instance.
      evaluator: Evaluator to calculate valid and test accuracies.
      rng: Numpy RNG from caller.
      experiment_metadata: experiment_data.ExperimentMetadata instance to append
        repeat data to.
      max_time_budget: Maximum time budget for search.
      mutation_rate: The probability to mutate a ModelSpec.
      population_size: The number of individuals in the population.
      save_child_history: Whether to save history of child. If True, its
        isomorphism-invariant graph hash, run indices, validation & test
        accuracies, and training times will be saved.
      search_method: Method of search: random, regularized_evolution, or
        elitism_evolution.
      tournament_size: The number of individuals selected for a tournament.
      repeat_idx: Index of experiment repeat.
  """
  print('Starting NASBench experiment repeat {}...'.format(repeat_idx))
  searcher = build_searcher(
      nasbench=nasbench,
      evaluator=evaluator,
      rng=rng,
      max_time_budget=max_time_budget,
      mutation_rate=mutation_rate,
      population_size=population_size,
      save_child_history=save_child_history,
      search_method=search_method,
      tournament_size=tournament_size,
  )

  statistics = None
  if searcher is not None:
    searcher.run_search()
    statistics = searcher.get_statistics()

  experiment_repeat_metadata = create_experiment_repeat_metadata_records(
      repeat_idx=repeat_idx, metadata_iterable_dict=statistics)
  experiment_metadata.repeat_metadata.append(experiment_repeat_metadata)
  print('NASBench experiment repeat {} complete.'.format(repeat_idx))


def run_experiment(
    nasbench: api.NASBench,
    experiment_id: str = '',
    experiment_repeat_indices: str = '0',
    fec_remove_probability: float = 0.0,
    fixed_run_idx: int = -1,
    hashing_time: float = 10.0,
    mantissa_bits: int = 24,
    max_num_evals: int = 1,
    max_time_budget: float = 5e6,
    mutation_rate: float = 1.0,
    noise_type: str = '',
    noise_stddev: float = 0.0,
    population_size: int = 50,
    save_child_history: bool = False,
    save_fec_history: bool = False,
    search_method: str = 'regularized_evolution',
    tournament_size: int = 10,
    use_fec: bool = False,
    rng_seed: Optional[int] = None) -> experiment_data.ExperimentMetadata:
  """Runs NASBench experiment.

  Args:
      nasbench: NASBench instance.
      experiment_id: The ID of the experiment to run.
      experiment_repeat_indices: Comma-separated string of experiment repeat
        indices.
      fec_remove_probability: After a cache hit, the probability a key will be
        removed from the cache.
      fixed_run_idx: The run index to use for getting computed_stats. If -1,
        then will be random and unfixed.
      hashing_time: Number of seconds it takes to generate hash.
      mantissa_bits: number of bits to use in the mantissa.
      max_num_evals: The maximum number of evaluations to perform for a hash.
      max_time_budget: Maximum time budget for search.
      mutation_rate: The probability to mutate a ModelSpec.
      noise_type: Type of noise to add to the evaluated validation accuracy.
      noise_stddev: Noise standard deviation for evaluated validation accuracy.
      population_size: The number of individuals in the population.
      save_child_history: Whether to save history of child. If True, its
        isomorphism-invariant graph hash, run indices, validation & test
        accuracies, and training times will be saved.
      save_fec_history: Whether to save FEC's history. If True, the FEC's number
        of cache misses, cache partial hits, cache full hits, cache size, and
        the Hasher's 4 accuracy model hash will be saved.
      search_method: Method of search: random, regularized_evolution, or
        elitism_evolution.
      tournament_size: The number of individuals selected for a tournament.
      use_fec: Whether to use FEC or not.
      rng_seed: Optional seed for RNG.

  Returns:
    ExperimentMetadata records for each repeat.
  """

  rng = np.random.default_rng(rng_seed)

  evaluator = evaluator_lib.Evaluator(
      nasbench=nasbench,
      fixed_run_idx=fixed_run_idx,
      mantissa_bits=mantissa_bits,
      hashing_time=hashing_time,
      noise_type=noise_type,
      noise_stddev=noise_stddev,
      use_fec=use_fec,
      max_num_evals=max_num_evals,
      fec_remove_probability=fec_remove_probability,
      save_fec_history=save_fec_history,
      rng_seed=rng.integers(low=1, high=constants.MAX_RNG_SEED, size=1)[0])
  repeat_indices = [int(x) for x in experiment_repeat_indices.split(',')]
  experiment_metadata = experiment_data.ExperimentMetadata(experiment_id)

  for repeat_idx in repeat_indices:
    run_experiment_repeat(
        nasbench=nasbench,
        evaluator=evaluator,
        rng=rng,
        experiment_metadata=experiment_metadata,
        max_time_budget=max_time_budget,
        mutation_rate=mutation_rate,
        population_size=population_size,
        save_child_history=save_child_history,
        search_method=search_method,
        tournament_size=tournament_size,
        repeat_idx=repeat_idx)

  return experiment_metadata
