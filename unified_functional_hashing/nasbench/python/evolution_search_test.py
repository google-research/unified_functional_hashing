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

"""Tests for evolution_search."""

import copy
import heapq
from typing import List, Tuple
from unittest import mock

from absl.testing import absltest
from unified_functional_hashing.nasbench.python import constants
from unified_functional_hashing.nasbench.python import evaluator as evaluator_lib
from unified_functional_hashing.nasbench.python import evolution_search
from unified_functional_hashing.nasbench.python import mutator as mutator_lib
from unified_functional_hashing.nasbench.python import random_spec_generator as random_spec_generator_lib
from unified_functional_hashing.nasbench.python import spec_validator_util
from unified_functional_hashing.nasbench.python import test_util
from unified_functional_hashing.nasbench.python import testing
from unified_functional_hashing.nasbench.python.testdata import computed_stats
from nasbench import api
import numpy as np

RNG_SEED = 42


def stringify_model_spec_tuple_matrices(
    model_spec_tuple: Tuple[Tuple[float, api.ModelSpec], ...]) -> str:
  """Converts tuple of tuples of ModelSpecs' matrices into string.

  Args:
    model_spec_tuple: Tuple of tuples of fitness-ModelSpec pairs.

  Returns:
    String representation of matrices.
  """
  return ";".join([
      ",".join([str(x)
                for x in indv[1].matrix.flatten().tolist()])
      for indv in model_spec_tuple
  ])


def stringify_model_spec_tuple_ops(
    model_spec_tuple: Tuple[Tuple[float, api.ModelSpec], ...]) -> str:
  """Converts tuple of tuples of ModelSpecs' ops into string.

  Args:
    model_spec_tuple: Tuple of tuples of fitness-ModelSpec pairs.

  Returns:
    String representation of ops.
  """
  return ";".join(
      [",".join([str(x) for x in indv[1].ops]) for indv in model_spec_tuple])


def stringify_model_spec_tuple(
    model_spec_tuple: Tuple[Tuple[float, api.ModelSpec],
                            ...]) -> Tuple[str, str]:
  """Converts tuple of tuples of ModelSpecs' ops into string.

  Args:
    model_spec_tuple: Tuple of tuples of fitness-ModelSpec pairs.

  Returns:
    Tuple string representation of matrices and ops.
  """
  matrix_str = stringify_model_spec_tuple_matrices(model_spec_tuple)
  ops_str = stringify_model_spec_tuple_ops(model_spec_tuple)
  return matrix_str, ops_str


def get_random_sample_matrix_and_ops(
    searcher: evolution_search.EvolutionSearcher,
    population: List[Tuple[float, api.ModelSpec]],
    tournament_size: int) -> Tuple[str, str]:
  """Gets random sample of ModelSpecs matrices and ops.

  Args:
    searcher: Searcher instance.
    population: List of fitness-ModelSpec tuples.
    tournament_size: The number of individuals selected for a tournament.

  Returns:
    Random sample's ModelSpec matrix string and ops string.
  """
  sample = searcher.random_combination(
      iterable=population, sample_size=tournament_size)
  return stringify_model_spec_tuple(sample)


def get_child_spec_matrix_and_ops(
    random_spec_generator: random_spec_generator_lib.RandomSpecGenerator,
    searcher: evolution_search.EvolutionSearcher) -> Tuple[str, str]:
  """Gets child ModelSpec matrix and ops.

  Args:
    random_spec_generator: RandomSpecGenerator instance.
    searcher: Searcher instance.

  Returns:
    Random ModelSpec matrix string and ops string.
  """
  parent_spec = random_spec_generator.random_spec()
  child_spec = searcher.get_child_spec(parent_spec=parent_spec)
  matrix_str_list = [str(x) for x in child_spec.matrix.flatten().tolist()]
  ops_str_list = child_spec.ops
  return ",".join(matrix_str_list), ",".join(ops_str_list)


class RandomSearchTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    mock_nasbench = mock.MagicMock()
    mock_nasbench.get_metrics_from_spec = mock.MagicMock(
        side_effect=test_util.get_metrics)
    mock_nasbench.config = constants.NASBENCH_CONFIG
    mock_nasbench.is_valid = mock.MagicMock(
        side_effect=spec_validator_util.is_valid)
    self._nasbench = mock_nasbench

  def test_add_random_individual_to_population(self):
    rng = np.random.default_rng(RNG_SEED)
    evaluator_rng_seed = rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    hashing_time = 10.0
    use_fec = False
    evaluator = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        mantissa_bits=24,
        hashing_time=hashing_time,
        noise_type="",
        noise_stddev=0.0,
        use_fec=use_fec,
        max_num_evals=1,
        fec_remove_probability=0.0,
        rng_seed=evaluator_rng_seed,
        test=True)

    max_time_budget = 2e4
    searcher_rng_seed = rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    searcher = evolution_search.EvolutionSearcher(
        nasbench=self._nasbench,
        evaluator=evaluator,
        max_time_budget=max_time_budget,
        population_size=6,
        tournament_size=2,
        mutation_rate=1.0,
        rng_seed=searcher_rng_seed)

    searcher.add_random_individual_to_population()

    statistics = searcher.get_statistics()
    population = searcher.get_population()

    # Calculate expected.
    expected_statistics = {
        "times": [0.0],
        "best_valids": [0.0],
        "best_tests": [0.0]
    }

    expected_rng = np.random.default_rng(RNG_SEED)
    expected_evaluator_rng_seed = expected_rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    expected_evaluator_rng = np.random.default_rng(expected_evaluator_rng_seed)
    expected_evaluator_rng2 = np.random.default_rng(expected_evaluator_rng_seed)
    _ = expected_evaluator_rng.integers(  # expected_noise_generator_seed
        low=1,
        high=constants.MAX_RNG_SEED,
        size=1)[0]
    _ = expected_evaluator_rng2.integers(  # expected_noise_generator_seed
        low=1,
        high=constants.MAX_RNG_SEED,
        size=1)[0]
    expected_searcher_rng = np.random.default_rng(searcher_rng_seed)
    expected_random_spec_generator_rng_seed = expected_searcher_rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    expected_random_spec_generator = (
        random_spec_generator_lib.RandomSpecGenerator(
            nasbench=self._nasbench,
            rng_seed=expected_random_spec_generator_rng_seed))

    expected_random_spec = expected_random_spec_generator.random_spec()
    graph_hash = expected_random_spec.hash_spec(
        canonical_ops=self._nasbench.config["nasbench_available_ops"])
    test_util.update_expected_iterators(
        expected_evaluator_rng,
        expected_statistics["times"],
        expected_statistics["best_valids"],
        expected_statistics["best_tests"],
        graph_hash=graph_hash,
        use_fec=use_fec,
        hashing_time=hashing_time)

    expected_computed_stats = computed_stats.COMPUTED_STATS[graph_hash]
    run_idx = expected_evaluator_rng2.integers(low=0, high=3, size=1)[0]
    expected_population = [
        (expected_computed_stats[108][run_idx]["final_validation_accuracy"],
         expected_random_spec)
    ]
    self.assertDictEqual(statistics, expected_statistics)
    self.assertLen(population, 1)
    self.assertListEqual([indv[0] for indv in population],
                         [indv[0] for indv in expected_population])
    self.assertListEqual(
        [indv[1].matrix.flatten().tolist() for indv in population],
        [indv[1].matrix.flatten().tolist() for indv in expected_population])
    self.assertListEqual([indv[1].ops for indv in population],
                         [indv[1].ops for indv in expected_population])

  # seed_population tests.
  def test_seed_population_exceed_max_budget(self):
    rng = np.random.default_rng(RNG_SEED)
    evaluator_rng_seed = rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    hashing_time = 10.0
    use_fec = False
    evaluator = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        noise_type="",
        noise_stddev=0.0,
        use_fec=use_fec,
        max_num_evals=1,
        fec_remove_probability=0.0,
        rng_seed=evaluator_rng_seed,
        test=True)

    max_time_budget = 1e4
    searcher_rng_seed = rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    searcher = evolution_search.EvolutionSearcher(
        nasbench=self._nasbench,
        evaluator=evaluator,
        max_time_budget=max_time_budget,
        population_size=3,
        tournament_size=1,
        mutation_rate=1.0,
        rng_seed=searcher_rng_seed)

    searcher.seed_population()

    population = searcher.get_population()
    statistics = searcher.get_statistics()

    # Calculate expected.
    expected_statistics = {
        "times": [0.0],
        "best_valids": [0.0],
        "best_tests": [0.0]
    }

    expected_rng = np.random.default_rng(RNG_SEED)
    expected_evaluator_rng_seed = expected_rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    expected_evaluator_rng = np.random.default_rng(expected_evaluator_rng_seed)
    expected_evaluator_rng2 = np.random.default_rng(expected_evaluator_rng_seed)
    _ = expected_evaluator_rng.integers(  # expected_noise_generator_seed
        low=1,
        high=constants.MAX_RNG_SEED,
        size=1)[0]
    _ = expected_evaluator_rng2.integers(  # expected_noise_generator_seed
        low=1,
        high=constants.MAX_RNG_SEED,
        size=1)[0]
    expected_searcher_rng = np.random.default_rng(searcher_rng_seed)
    expected_random_spec_generator_rng_seed = expected_searcher_rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    expected_random_spec_generator = (
        random_spec_generator_lib.RandomSpecGenerator(
            nasbench=self._nasbench,
            rng_seed=expected_random_spec_generator_rng_seed))

    expected_random_spec = expected_random_spec_generator.random_spec()
    graph_hash = expected_random_spec.hash_spec(
        canonical_ops=self._nasbench.config["nasbench_available_ops"])
    test_util.update_expected_iterators(
        expected_evaluator_rng,
        expected_statistics["times"],
        expected_statistics["best_valids"],
        expected_statistics["best_tests"],
        graph_hash=graph_hash,
        use_fec=use_fec,
        hashing_time=hashing_time)

    expected_computed_stats = computed_stats.COMPUTED_STATS[graph_hash]
    run_idx = expected_evaluator_rng2.integers(low=0, high=3, size=1)[0]
    expected_population = [
        (expected_computed_stats[108][run_idx]["final_validation_accuracy"],
         expected_random_spec)
    ]
    self.assertDictEqual(statistics, expected_statistics)
    self.assertLen(population, 1)
    self.assertListEqual([indv[0] for indv in population],
                         [indv[0] for indv in expected_population])
    self.assertListEqual(
        [indv[1].matrix.flatten().tolist() for indv in population],
        [indv[1].matrix.flatten().tolist() for indv in expected_population])
    self.assertListEqual([indv[1].ops for indv in population],
                         [indv[1].ops for indv in expected_population])

  def test_seed_population_within_max_budget(self):
    rng = np.random.default_rng(RNG_SEED)
    evaluator_rng_seed = rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    hashing_time = 10.0
    use_fec = False
    evaluator = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        mantissa_bits=24,
        hashing_time=hashing_time,
        noise_type="",
        noise_stddev=0.0,
        use_fec=use_fec,
        max_num_evals=1,
        fec_remove_probability=0.0,
        rng_seed=evaluator_rng_seed,
        test=True)

    population_size = 3
    max_time_budget = 2e4
    searcher_rng_seed = rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    searcher = evolution_search.EvolutionSearcher(
        nasbench=self._nasbench,
        evaluator=evaluator,
        evolution_type="regularized_evolution",
        max_time_budget=max_time_budget,
        population_size=population_size,
        tournament_size=1,
        mutation_rate=1.0,
        rng_seed=searcher_rng_seed)

    searcher.seed_population()

    population = searcher.get_population()
    statistics = searcher.get_statistics()

    # Calculate expected.
    expected_statistics = {
        "times": [0.0],
        "best_valids": [0.0],
        "best_tests": [0.0]
    }

    expected_population = []

    expected_rng = np.random.default_rng(RNG_SEED)
    expected_evaluator_rng_seed = expected_rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    expected_evaluator_rng = np.random.default_rng(expected_evaluator_rng_seed)
    expected_evaluator_rng2 = np.random.default_rng(expected_evaluator_rng_seed)
    _ = expected_evaluator_rng.integers(  # expected_noise_generator_seed
        low=1,
        high=constants.MAX_RNG_SEED,
        size=1)[0]
    _ = expected_evaluator_rng2.integers(  # expected_noise_generator_seed
        low=1,
        high=constants.MAX_RNG_SEED,
        size=1)[0]
    expected_searcher_rng = np.random.default_rng(searcher_rng_seed)
    expected_random_spec_generator_rng_seed = expected_searcher_rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    expected_random_spec_generator = (
        random_spec_generator_lib.RandomSpecGenerator(
            nasbench=self._nasbench,
            rng_seed=expected_random_spec_generator_rng_seed))

    while (statistics["times"][-1] < max_time_budget and
           len(expected_population) < population_size):
      expected_random_spec = expected_random_spec_generator.random_spec()
      graph_hash = expected_random_spec.hash_spec(
          canonical_ops=self._nasbench.config["nasbench_available_ops"])
      test_util.update_expected_iterators(
          expected_evaluator_rng,
          expected_statistics["times"],
          expected_statistics["best_valids"],
          expected_statistics["best_tests"],
          graph_hash=graph_hash,
          use_fec=use_fec,
          hashing_time=hashing_time)

      expected_computed_stats = computed_stats.COMPUTED_STATS[graph_hash]
      run_idx = expected_evaluator_rng2.integers(low=0, high=3, size=1)[0]
      expected_population.append(
          (expected_computed_stats[108][run_idx]["final_validation_accuracy"],
           expected_random_spec))

    self.assertDictEqual(statistics, expected_statistics)
    self.assertLen(population, population_size)
    self.assertListEqual([indv[0] for indv in population],
                         [indv[0] for indv in expected_population])
    self.assertListEqual(
        [indv[1].matrix.flatten().tolist() for indv in population],
        [indv[1].matrix.flatten().tolist() for indv in expected_population])
    self.assertListEqual([indv[1].ops for indv in population],
                         [indv[1].ops for indv in expected_population])

  def test_random_combination(self):
    rng = np.random.default_rng(RNG_SEED)
    evaluator_rng_seed = rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    hashing_time = 10.0
    evaluator = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        mantissa_bits=24,
        hashing_time=hashing_time,
        noise_type="",
        noise_stddev=0.0,
        use_fec=False,
        max_num_evals=1,
        fec_remove_probability=0.0,
        rng_seed=evaluator_rng_seed,
        test=True)

    population_size = 5
    tournament_size = 2
    searcher_rng_seed = rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    searcher = evolution_search.EvolutionSearcher(
        nasbench=self._nasbench,
        evaluator=evaluator,
        evolution_type="regularized_evolution",
        max_time_budget=2e4,
        population_size=population_size,
        tournament_size=tournament_size,
        mutation_rate=1.0,
        rng_seed=searcher_rng_seed)

    random_spec_generator = random_spec_generator_lib.RandomSpecGenerator(
        nasbench=self._nasbench, rng_seed=RNG_SEED)
    population = [(0.0, random_spec_generator.random_spec())
                  for _ in range(population_size)]
    expected_sample = tuple(population[:tournament_size])
    expected_sample_str = stringify_model_spec_tuple(expected_sample)
    self.assertTrue(
        testing.is_eventually(
            func=lambda: get_random_sample_matrix_and_ops(  # pylint: disable=g-long-lambda
                searcher=searcher,
                population=population,
                tournament_size=tournament_size),
            required_values=[expected_sample_str],
            allowed_values=None,
            max_run_time=30.0))

  def test_best_spec_from_sample(self):
    rng = np.random.default_rng(RNG_SEED)
    evaluator_rng_seed = rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    hashing_time = 10.0
    evaluator = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        mantissa_bits=24,
        hashing_time=hashing_time,
        noise_type="",
        noise_stddev=0.0,
        use_fec=False,
        max_num_evals=1,
        fec_remove_probability=0.0,
        rng_seed=evaluator_rng_seed,
        test=True)

    population_size = 100
    tournament_size = 10
    searcher_rng_seed = rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    searcher = evolution_search.EvolutionSearcher(
        nasbench=self._nasbench,
        evaluator=evaluator,
        evolution_type="regularized_evolution",
        max_time_budget=2e4,
        population_size=population_size,
        tournament_size=tournament_size,
        mutation_rate=1.0,
        rng_seed=searcher_rng_seed)

    random_spec_generator = random_spec_generator_lib.RandomSpecGenerator(
        nasbench=self._nasbench, rng_seed=RNG_SEED)
    population = [(rng.random(size=1)[0], random_spec_generator.random_spec())
                  for _ in range(population_size)]
    # Modify tournament // 2 index to have the highest fitness.
    expected_best_spec = population[tournament_size // 2][1]
    population[tournament_size // 2] = (12345.0, expected_best_spec)
    expected_sample = tuple(population[:tournament_size])

    best_spec = searcher.get_best_spec_from_sample(sample=expected_sample)

    self.assertListEqual(best_spec.matrix.flatten().tolist(),
                         expected_best_spec.matrix.flatten().tolist())
    self.assertListEqual(best_spec.ops, expected_best_spec.ops)

  def test_best_spec_from_tournament(self):
    rng = np.random.default_rng(RNG_SEED)
    evaluator_rng_seed = rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    hashing_time = 10.0
    evaluator = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        mantissa_bits=24,
        hashing_time=hashing_time,
        noise_type="",
        noise_stddev=0.0,
        use_fec=False,
        max_num_evals=1,
        fec_remove_probability=0.0,
        rng_seed=evaluator_rng_seed,
        test=True)

    population_size = 100
    tournament_size = 10
    searcher_rng_seed = rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]

    random_spec_generator = random_spec_generator_lib.RandomSpecGenerator(
        nasbench=self._nasbench, rng_seed=RNG_SEED)
    population = [(rng.random(size=1)[0], random_spec_generator.random_spec())
                  for _ in range(population_size)]
    # Modify tournament // 2 index to have the highest fitness.
    expected_best_spec = population[tournament_size // 2][1]
    population[tournament_size // 2] = (12345.0, expected_best_spec)

    searcher = evolution_search.EvolutionSearcher(
        nasbench=self._nasbench,
        evaluator=evaluator,
        evolution_type="regularized_evolution",
        max_time_budget=2e4,
        population_size=population_size,
        tournament_size=tournament_size,
        mutation_rate=1.0,
        rng_seed=searcher_rng_seed,
        population=population)

    best_spec = searcher.get_best_spec_from_tournament()

    self.assertListEqual(best_spec.matrix.flatten().tolist(),
                         expected_best_spec.matrix.flatten().tolist())
    self.assertListEqual(best_spec.ops, expected_best_spec.ops)

  def test_get_child_spec_no_fim(self):
    rng = np.random.default_rng(RNG_SEED)
    evaluator_rng_seed = rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    hashing_time = 10.0
    evaluator = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        mantissa_bits=24,
        hashing_time=hashing_time,
        noise_type="",
        noise_stddev=0.0,
        use_fec=False,
        max_num_evals=1,
        fec_remove_probability=0.0,
        rng_seed=evaluator_rng_seed,
        test=True)

    searcher_rng_seed = rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    searcher = evolution_search.EvolutionSearcher(
        nasbench=self._nasbench,
        evaluator=evaluator,
        evolution_type="regularized_evolution",
        max_time_budget=2e4,
        population_size=100,
        tournament_size=10,
        mutation_rate=1.0,
        rng_seed=searcher_rng_seed)

    random_spec_generator = random_spec_generator_lib.RandomSpecGenerator(
        nasbench=self._nasbench, rng_seed=RNG_SEED)

    expected_matrix = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
    expected_ops = ["input", "maxpool3x3", "output"]

    self.assertTrue(
        testing.is_eventually(
            func=lambda: get_child_spec_matrix_and_ops(  # pylint: disable=g-long-lambda
                random_spec_generator, searcher),
            required_values=[
                (",".join([str(x) for x in expected_matrix.flatten().tolist()]),
                 ",".join(expected_ops))
            ],
            allowed_values=None,
            max_run_time=30.0))

  # update_population tests.
  def test_update_population_regularized_evolution(self):
    rng = np.random.default_rng(RNG_SEED)
    evaluator_rng_seed = rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    hashing_time = 10.0
    evaluator = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        mantissa_bits=24,
        hashing_time=hashing_time,
        noise_type="",
        noise_stddev=0.0,
        use_fec=False,
        max_num_evals=1,
        fec_remove_probability=0.0,
        rng_seed=evaluator_rng_seed,
        test=True)

    population_size = 100
    searcher_rng_seed = rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]

    random_spec_generator = random_spec_generator_lib.RandomSpecGenerator(
        nasbench=self._nasbench, rng_seed=RNG_SEED)
    population = [(rng.random(size=1)[0], random_spec_generator.random_spec())
                  for _ in range(population_size)]

    searcher = evolution_search.EvolutionSearcher(
        nasbench=self._nasbench,
        evaluator=evaluator,
        evolution_type="regularized_evolution",
        max_time_budget=2e4,
        population_size=population_size,
        tournament_size=10,
        mutation_rate=1.0,
        rng_seed=searcher_rng_seed,
        population=population)

    new_spec = random_spec_generator.random_spec()
    eval_outputs = (0, 0.42, 0.314, 1500.0)  # run_idx, valid, test, time
    expected_valid_fitnesses = [indv[0] for indv in population[1:]
                               ] + [eval_outputs[1]]
    expected_matrices = [
        indv[1].matrix.flatten().tolist() for indv in population[1:]
    ] + [new_spec.matrix.flatten().tolist()]
    expected_ops = [indv[1].ops for indv in population[1:]] + [new_spec.ops]

    searcher.update_population(
        new_spec=new_spec, eval_outputs=eval_outputs, population_heap=None)

    self.assertListEqual([indv[0] for indv in population],
                         expected_valid_fitnesses)
    self.assertListEqual(
        [indv[1].matrix.flatten().tolist() for indv in population],
        expected_matrices)
    self.assertListEqual([indv[1].ops for indv in population], expected_ops)

  def test_update_population_elitism_evolution(self):
    rng = np.random.default_rng(RNG_SEED)
    evaluator_rng_seed = rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    hashing_time = 10.0
    evaluator = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        mantissa_bits=24,
        hashing_time=hashing_time,
        noise_type="",
        noise_stddev=0.0,
        use_fec=False,
        max_num_evals=1,
        fec_remove_probability=0.0,
        rng_seed=evaluator_rng_seed,
        test=True)

    population_size = 100
    searcher_rng_seed = rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]

    random_spec_generator = random_spec_generator_lib.RandomSpecGenerator(
        nasbench=self._nasbench, rng_seed=RNG_SEED)
    population = [(rng.random(size=1)[0], random_spec_generator.random_spec())
                  for _ in range(population_size)]
    # Make population_size // 2 individual have worst fitness so it's killed.
    population[population_size // 2] = (-12345.0,
                                        population[population_size // 2][1])

    searcher = evolution_search.EvolutionSearcher(
        nasbench=self._nasbench,
        evaluator=evaluator,
        evolution_type="elitism_evolution",
        max_time_budget=2e4,
        population_size=population_size,
        tournament_size=10,
        mutation_rate=1.0,
        rng_seed=searcher_rng_seed,
        population=population)

    population_heap = [(fitness, i) for i, (fitness, _) in enumerate(population)
                      ]
    heapq.heapify(population_heap)

    new_spec = random_spec_generator.random_spec()
    eval_outputs = (0, 0.42, 0.314, 1500.0)  # run_idx, valid, test, time
    expected_valid_fitnesses = [
        indv[0] for indv in population[:population_size // 2]
    ] + [eval_outputs[1]
        ] + [indv[0] for indv in population[population_size // 2 + 1:]]
    expected_matrices = [
        indv[1].matrix.flatten().tolist()
        for indv in population[:population_size // 2]
    ] + [new_spec.matrix.flatten().tolist()] + [
        indv[1].matrix.flatten().tolist()
        for indv in population[population_size // 2 + 1:]
    ]
    expected_ops = [
        indv[1].ops for indv in population[:population_size // 2]
    ] + [new_spec.ops
        ] + [indv[1].ops for indv in population[population_size // 2 + 1:]]

    expected_population_heap = copy.deepcopy(population_heap)
    _, worst_idx = heapq.heappop(expected_population_heap)
    heapq.heappush(expected_population_heap, (eval_outputs[1], worst_idx))

    searcher.update_population(
        new_spec=new_spec,
        eval_outputs=eval_outputs,
        population_heap=population_heap)

    self.assertListEqual([indv[0] for indv in population],
                         expected_valid_fitnesses)
    self.assertListEqual(
        [indv[1].matrix.flatten().tolist() for indv in population],
        expected_matrices)
    self.assertListEqual([indv[1].ops for indv in population], expected_ops)
    self.assertListEqual([indv[0] for indv in population_heap],
                         [indv[0] for indv in expected_population_heap])
    self.assertListEqual([indv[1] for indv in population_heap],
                         [indv[1] for indv in expected_population_heap])

  def test_update_population_wrong_type(self):
    rng = np.random.default_rng(RNG_SEED)
    evaluator_rng_seed = rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    hashing_time = 10.0
    evaluator = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        mantissa_bits=24,
        hashing_time=hashing_time,
        noise_type="",
        noise_stddev=0.0,
        use_fec=False,
        max_num_evals=1,
        fec_remove_probability=0.0,
        rng_seed=evaluator_rng_seed,
        test=True)

    population_size = 100
    searcher_rng_seed = rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]

    random_spec_generator = random_spec_generator_lib.RandomSpecGenerator(
        nasbench=self._nasbench, rng_seed=RNG_SEED)
    population = [(rng.random(size=1)[0], random_spec_generator.random_spec())
                  for _ in range(population_size)]

    evolution_type = "bad_type"
    searcher = evolution_search.EvolutionSearcher(
        nasbench=self._nasbench,
        evaluator=evaluator,
        evolution_type=evolution_type,
        max_time_budget=2e4,
        population_size=population_size,
        tournament_size=10,
        mutation_rate=1.0,
        rng_seed=searcher_rng_seed,
        population=population)

    new_spec = random_spec_generator.random_spec()
    eval_outputs = (0, 0.42, 0.314, 1500.0)  # run_idx, valid, test, time

    with self.assertRaisesRegex(
        expected_exception=ValueError,
        expected_regex="{} is not a valid evolution_type!".format(
            evolution_type)):
      searcher.update_population(
          new_spec=new_spec, eval_outputs=eval_outputs, population_heap=None)

  # perform_search_iteration tests.
  def test_perform_search_iteration_regularized_evolution_no_fim_no_fec(self):
    rng = np.random.default_rng(RNG_SEED)
    evaluator_rng_seed = rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    evaluator_rng = np.random.default_rng(evaluator_rng_seed)
    _ = evaluator_rng.integers(  # evaluator_noise_generator_rng_seed
        low=1,
        high=constants.MAX_RNG_SEED,
        size=1)[0]
    hashing_time = 10.0
    evaluator = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        mantissa_bits=24,
        hashing_time=hashing_time,
        noise_type="",
        noise_stddev=0.0,
        use_fec=False,
        max_num_evals=1,
        fec_remove_probability=0.0,
        rng_seed=evaluator_rng_seed,
        test=True)

    population_size = 5
    searcher_rng_seed = rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]

    random_spec_generator = random_spec_generator_lib.RandomSpecGenerator(
        nasbench=self._nasbench, rng_seed=RNG_SEED)
    population = [(rng.random(size=1)[0], random_spec_generator.random_spec())
                  for _ in range(population_size)]
    # Make population_size // 2 have highest fitness so that wins tournament.
    expected_best_spec = population[population_size // 2][1]
    population[population_size // 2] = (12345.0, expected_best_spec)

    searcher = evolution_search.EvolutionSearcher(
        nasbench=self._nasbench,
        evaluator=evaluator,
        evolution_type="regularized_evolution",
        max_time_budget=2e4,
        population_size=population_size,
        tournament_size=3,
        mutation_rate=1.0,
        rng_seed=searcher_rng_seed,
        population=population)

    expected_searcher_rng = np.random.default_rng(searcher_rng_seed)
    _ = expected_searcher_rng.integers(  # random_spec_generator_rng_seed
        low=1,
        high=constants.MAX_RNG_SEED,
        size=1)[0]
    expected_mutator_rng_seed = expected_searcher_rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    mutator = mutator_lib.Mutator(
        nasbench=self._nasbench,
        mutation_rate=1.0,
        rng_seed=expected_mutator_rng_seed)
    expected_child_spec = mutator.mutate_spec(old_spec=expected_best_spec)
    graph_hash = expected_child_spec.hash_spec(
        canonical_ops=self._nasbench.config["nasbench_available_ops"])
    run_idx = evaluator_rng.integers(low=0, high=3, size=1)[0]
    stats = computed_stats.COMPUTED_STATS[graph_hash][108][run_idx]
    eval_outputs = (stats["final_validation_accuracy"],
                    stats["final_test_accuracy"], stats["final_training_time"])

    expected_statistics = {
        "times": [0.0, eval_outputs[-1]],
        "best_valids": [0.0, eval_outputs[0]],
        "best_tests": [0.0, eval_outputs[1]]
    }

    expected_valid_fitnesses = [indv[0] for indv in population[1:]
                               ] + [eval_outputs[0]]
    expected_matrices = [
        indv[1].matrix.flatten().tolist() for indv in population[1:]
    ] + [expected_child_spec.matrix.flatten().tolist()]
    expected_ops = [indv[1].ops for indv in population[1:]
                   ] + [expected_child_spec.ops]

    searcher.perform_search_iteration()

    statistics = searcher.get_statistics()

    self.assertDictEqual(statistics, expected_statistics)

    self.assertListEqual([indv[0] for indv in population],
                         expected_valid_fitnesses)
    self.assertListEqual(
        [indv[1].matrix.flatten().tolist() for indv in population],
        expected_matrices)
    self.assertListEqual([indv[1].ops for indv in population], expected_ops)

  def test_perform_search_iteration_elitism_evolution_no_fim_no_fec(self):
    rng = np.random.default_rng(RNG_SEED)
    evaluator_rng_seed = rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    evaluator_rng = np.random.default_rng(evaluator_rng_seed)
    _ = evaluator_rng.integers(  # evaluator_noise_generator_rng_seed
        low=1,
        high=constants.MAX_RNG_SEED,
        size=1)[0]
    hashing_time = 10.0
    evaluator = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        mantissa_bits=24,
        hashing_time=hashing_time,
        noise_type="",
        noise_stddev=0.0,
        use_fec=False,
        max_num_evals=1,
        fec_remove_probability=0.0,
        rng_seed=evaluator_rng_seed,
        test=True)

    population_size = 5
    searcher_rng_seed = rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]

    random_spec_generator = random_spec_generator_lib.RandomSpecGenerator(
        nasbench=self._nasbench, rng_seed=RNG_SEED)
    population = [(rng.random(size=1)[0], random_spec_generator.random_spec())
                  for _ in range(population_size)]
    # Make population_size // 2 individual have worst fitness so it's killed.
    expected_worst_spec = population[population_size // 2][1]
    population[population_size // 2] = (-12345.0, expected_worst_spec)

    # Make population_size // 2 - 1 individual have best fitness so it wins
    # tournament.
    expected_best_spec = population[population_size // 2 - 1][1]
    population[population_size // 2 - 1] = (12345.0, expected_best_spec)

    searcher = evolution_search.EvolutionSearcher(
        nasbench=self._nasbench,
        evaluator=evaluator,
        evolution_type="elitism_evolution",
        max_time_budget=2e4,
        population_size=population_size,
        tournament_size=3,
        mutation_rate=1.0,
        rng_seed=searcher_rng_seed,
        population=population)

    population_heap = [(fitness, i) for i, (fitness, _) in enumerate(population)
                      ]
    heapq.heapify(population_heap)

    expected_searcher_rng = np.random.default_rng(searcher_rng_seed)
    _ = expected_searcher_rng.integers(  # random_spec_generator_rng_seed
        low=1,
        high=constants.MAX_RNG_SEED,
        size=1)[0]
    expected_mutator_rng_seed = expected_searcher_rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    mutator = mutator_lib.Mutator(
        nasbench=self._nasbench,
        mutation_rate=1.0,
        rng_seed=expected_mutator_rng_seed)
    expected_child_spec = mutator.mutate_spec(old_spec=expected_best_spec)
    graph_hash = expected_child_spec.hash_spec(
        canonical_ops=self._nasbench.config["nasbench_available_ops"])
    run_idx = evaluator_rng.integers(low=0, high=3, size=1)[0]
    stats = computed_stats.COMPUTED_STATS[graph_hash][108][run_idx]
    eval_outputs = (stats["final_validation_accuracy"],
                    stats["final_test_accuracy"], stats["final_training_time"])
    expected_statistics = {
        "times": [0.0, eval_outputs[-1]],
        "best_valids": [0.0, eval_outputs[0]],
        "best_tests": [0.0, eval_outputs[1]]
    }

    expected_population = population[:population_size // 2] + [
        (eval_outputs[0], expected_child_spec)
    ] + population[population_size // 2 + 1:]
    expected_valid_fitnesses = [indv[0] for indv in expected_population]
    expected_matrices = [
        indv[1].matrix.flatten().tolist() for indv in expected_population
    ]
    expected_ops = [indv[1].ops for indv in expected_population]

    searcher.perform_search_iteration(population_heap=population_heap)

    statistics = searcher.get_statistics()

    self.assertDictEqual(statistics, expected_statistics)

    self.assertListEqual([indv[0] for indv in population],
                         expected_valid_fitnesses)
    self.assertListEqual(
        [indv[1].matrix.flatten().tolist() for indv in population],
        expected_matrices)
    self.assertListEqual([indv[1].ops for indv in population], expected_ops)

  def test_perform_search_iteration_regularized_evolution_no_fim_fec(self):
    rng = np.random.default_rng(RNG_SEED)
    evaluator_rng_seed = rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    evaluator_rng = np.random.default_rng(evaluator_rng_seed)
    _ = evaluator_rng.integers(  # evaluator_noise_generator_rng_seed
        low=1,
        high=constants.MAX_RNG_SEED,
        size=1)[0]
    evaluator_fec_rng_seed = evaluator_rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    evaluator_fec_rng = np.random.default_rng(evaluator_fec_rng_seed)
    hashing_time = 10.0
    evaluator = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        mantissa_bits=24,
        hashing_time=hashing_time,
        noise_type="",
        noise_stddev=0.0,
        use_fec=True,
        max_num_evals=1,
        fec_remove_probability=0.0,
        save_fec_history=True,
        rng_seed=evaluator_rng_seed,
        test=True)

    population_size = 5
    searcher_rng_seed = rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]

    random_spec_generator = random_spec_generator_lib.RandomSpecGenerator(
        nasbench=self._nasbench, rng_seed=RNG_SEED)
    population = [(rng.random(size=1)[0], random_spec_generator.random_spec())
                  for _ in range(population_size)]
    # Make population_size // 2 have highest fitness so that wins tournament.
    expected_best_spec = population[population_size // 2][1]
    population[population_size // 2] = (12345.0, expected_best_spec)

    searcher = evolution_search.EvolutionSearcher(
        nasbench=self._nasbench,
        evaluator=evaluator,
        evolution_type="regularized_evolution",
        max_time_budget=2e4,
        population_size=population_size,
        tournament_size=3,
        mutation_rate=1.0,
        rng_seed=searcher_rng_seed,
        population=population)

    expected_searcher_rng = np.random.default_rng(searcher_rng_seed)
    _ = expected_searcher_rng.integers(  # random_spec_generator_rng_seed
        low=1,
        high=constants.MAX_RNG_SEED,
        size=1)[0]
    expected_mutator_rng_seed = expected_searcher_rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    mutator = mutator_lib.Mutator(
        nasbench=self._nasbench,
        mutation_rate=1.0,
        rng_seed=expected_mutator_rng_seed)
    expected_child_spec = mutator.mutate_spec(old_spec=expected_best_spec)
    graph_hash = expected_child_spec.hash_spec(
        canonical_ops=self._nasbench.config["nasbench_available_ops"])
    run_idx = evaluator_fec_rng.integers(low=0, high=3, size=1)[0]
    stats = computed_stats.COMPUTED_STATS[graph_hash][108][run_idx]
    eval_outputs = (stats["final_validation_accuracy"],
                    stats["final_test_accuracy"],
                    stats["final_training_time"] + hashing_time)

    expected_statistics = {
        "times": [0.0, eval_outputs[-1]],
        "best_valids": [0.0, eval_outputs[0]],
        "best_tests": [0.0, eval_outputs[1]],
        "num_cache_misses": [0, 1],
        "num_cache_partial_hits": [0, 0],
        "num_cache_full_hits": [0, 0],
        "cache_size": [0, 1],
        "model_hashes": ["", "3380622801569494190"]
    }

    expected_valid_fitnesses = [indv[0] for indv in population[1:]
                               ] + [eval_outputs[0]]
    expected_matrices = [
        indv[1].matrix.flatten().tolist() for indv in population[1:]
    ] + [expected_child_spec.matrix.flatten().tolist()]
    expected_ops = [indv[1].ops for indv in population[1:]
                   ] + [expected_child_spec.ops]

    searcher.perform_search_iteration()

    statistics = searcher.get_statistics()

    self.assertDictEqual(statistics, expected_statistics)

    self.assertListEqual([indv[0] for indv in population],
                         expected_valid_fitnesses)
    self.assertListEqual(
        [indv[1].matrix.flatten().tolist() for indv in population],
        expected_matrices)
    self.assertListEqual([indv[1].ops for indv in population], expected_ops)

  def test_perform_search_iteration_elitism_evolution_no_fim_fec(self):
    rng = np.random.default_rng(RNG_SEED)
    evaluator_rng_seed = rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    evaluator_rng = np.random.default_rng(evaluator_rng_seed)
    _ = evaluator_rng.integers(  # evaluator_noise_generator_rng_seed
        low=1,
        high=constants.MAX_RNG_SEED,
        size=1)[0]
    evaluator_fec_rng_seed = evaluator_rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    evaluator_fec_rng = np.random.default_rng(evaluator_fec_rng_seed)
    hashing_time = 10.0
    evaluator = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        mantissa_bits=24,
        hashing_time=hashing_time,
        noise_type="",
        noise_stddev=0.0,
        use_fec=True,
        max_num_evals=1,
        fec_remove_probability=0.0,
        save_fec_history=True,
        rng_seed=evaluator_rng_seed,
        test=True)

    population_size = 5
    searcher_rng_seed = rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]

    random_spec_generator = random_spec_generator_lib.RandomSpecGenerator(
        nasbench=self._nasbench, rng_seed=RNG_SEED)
    population = [(rng.random(size=1)[0], random_spec_generator.random_spec())
                  for _ in range(population_size)]
    # Make population_size // 2 individual have worst fitness so it's killed.
    expected_worst_spec = population[population_size // 2][1]
    population[population_size // 2] = (-12345.0, expected_worst_spec)

    # Make population_size // 2 - 1 individual have best fitness so it wins
    # tournament.
    expected_best_spec = population[population_size // 2 - 1][1]
    population[population_size // 2 - 1] = (12345.0, expected_best_spec)

    searcher = evolution_search.EvolutionSearcher(
        nasbench=self._nasbench,
        evaluator=evaluator,
        evolution_type="elitism_evolution",
        max_time_budget=2e4,
        population_size=population_size,
        tournament_size=3,
        mutation_rate=1.0,
        rng_seed=searcher_rng_seed,
        population=population)

    population_heap = [(fitness, i) for i, (fitness, _) in enumerate(population)
                      ]
    heapq.heapify(population_heap)

    expected_searcher_rng = np.random.default_rng(searcher_rng_seed)
    _ = expected_searcher_rng.integers(  # random_spec_generator_rng_seed
        low=1,
        high=constants.MAX_RNG_SEED,
        size=1)[0]
    expected_mutator_rng_seed = expected_searcher_rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    mutator = mutator_lib.Mutator(
        nasbench=self._nasbench,
        mutation_rate=1.0,
        rng_seed=expected_mutator_rng_seed)
    expected_child_spec = mutator.mutate_spec(old_spec=expected_best_spec)
    graph_hash = expected_child_spec.hash_spec(
        canonical_ops=self._nasbench.config["nasbench_available_ops"])
    run_idx = evaluator_fec_rng.integers(low=0, high=3, size=1)[0]
    stats = computed_stats.COMPUTED_STATS[graph_hash][108][run_idx]
    eval_outputs = (stats["final_validation_accuracy"],
                    stats["final_test_accuracy"],
                    stats["final_training_time"] + hashing_time)
    expected_statistics = {
        "times": [0.0, eval_outputs[-1]],
        "best_valids": [0.0, eval_outputs[0]],
        "best_tests": [0.0, eval_outputs[1]],
        "num_cache_misses": [0, 1],
        "num_cache_partial_hits": [0, 0],
        "num_cache_full_hits": [0, 0],
        "cache_size": [0, 1],
        "model_hashes": ["", "6128025405464510464"]
    }

    expected_population = population[:population_size // 2] + [
        (eval_outputs[0], expected_child_spec)
    ] + population[population_size // 2 + 1:]
    expected_valid_fitnesses = [indv[0] for indv in expected_population]
    expected_matrices = [
        indv[1].matrix.flatten().tolist() for indv in expected_population
    ]
    expected_ops = [indv[1].ops for indv in expected_population]

    searcher.perform_search_iteration(population_heap=population_heap)

    statistics = searcher.get_statistics()

    self.assertDictEqual(statistics, expected_statistics)

    self.assertListEqual([indv[0] for indv in population],
                         expected_valid_fitnesses)
    self.assertListEqual(
        [indv[1].matrix.flatten().tolist() for indv in population],
        expected_matrices)
    self.assertListEqual([indv[1].ops for indv in population], expected_ops)

  # evolution_search tests.
  def test_evolution_search_regularized_evolution_no_fim_no_fec(self):
    rng = np.random.default_rng(RNG_SEED)
    evaluator_rng_seed = rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    evaluator_rng = np.random.default_rng(evaluator_rng_seed)
    _ = evaluator_rng.integers(  # evaluator_noise_generator_rng_seed
        low=1,
        high=constants.MAX_RNG_SEED,
        size=1)[0]
    hashing_time = 10.0
    evaluator = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        mantissa_bits=24,
        hashing_time=hashing_time,
        noise_type="",
        noise_stddev=0.0,
        use_fec=False,
        max_num_evals=1,
        fec_remove_probability=0.0,
        rng_seed=evaluator_rng_seed,
        test=True)

    searcher_rng_seed = rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    searcher = evolution_search.EvolutionSearcher(
        nasbench=self._nasbench,
        evaluator=evaluator,
        evolution_type="regularized_evolution",
        max_time_budget=2e4,
        population_size=5,
        tournament_size=3,
        mutation_rate=1.0,
        rng_seed=searcher_rng_seed)

    searcher.run_search()

    statistics = searcher.get_statistics()

    expected_times = [
        0.0, 11155.85302734375, 12681.709126524445, 14193.926641386897,
        15640.971124401605, 17206.752775833298, 18756.852652890677,
        20373.39940139911
    ]
    expected_best_valids = [
        0.0, 0.19376001358032227, 0.6863627939498267, 0.6863627939498267,
        0.6863627939498267, 0.6965063927814837, 0.6965063927814837,
        0.6965063927814837
    ]
    expected_best_tests = [
        0.0, 0.19311898946762085, 0.7232977219136525, 0.7232977219136525,
        0.7232977219136525, 0.7025721170487348, 0.7025721170487348,
        0.7025721170487348
    ]

    expected_statistics = {
        "times": expected_times,
        "best_valids": expected_best_valids,
        "best_tests": expected_best_tests
    }

    self.assertDictEqual(statistics, expected_statistics)

  def test_evolution_search_regularized_evolution_no_fim_fec(self):
    rng = np.random.default_rng(RNG_SEED)
    evaluator_rng_seed = rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    evaluator_rng = np.random.default_rng(evaluator_rng_seed)
    _ = evaluator_rng.integers(  # evaluator_noise_generator_rng_seed
        low=1,
        high=constants.MAX_RNG_SEED,
        size=1)[0]
    mantissa_bits = 24
    hashing_time = 10.0
    noise_type = ""
    noise_stddev = 0.0
    use_fec = True
    max_num_evals = 1
    fec_remove_probability = 0.0
    save_fec_history = True
    evaluator = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        mantissa_bits=mantissa_bits,
        hashing_time=hashing_time,
        noise_type=noise_type,
        noise_stddev=noise_stddev,
        use_fec=use_fec,
        max_num_evals=max_num_evals,
        fec_remove_probability=fec_remove_probability,
        save_fec_history=save_fec_history,
        rng_seed=evaluator_rng_seed,
        test=True)

    searcher_rng_seed = rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    evolution_type = "regularized_evolution"
    max_time_budget = 2e4
    population_size = 5
    tournament_size = 3
    mutation_rate = 1.0
    searcher = evolution_search.EvolutionSearcher(
        nasbench=self._nasbench,
        evaluator=evaluator,
        evolution_type=evolution_type,
        max_time_budget=max_time_budget,
        population_size=population_size,
        tournament_size=tournament_size,
        mutation_rate=mutation_rate,
        rng_seed=searcher_rng_seed)

    searcher.run_search()

    # Force cache hits by reinitializing evaluator and copying FEC cache.
    evaluator2 = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        mantissa_bits=mantissa_bits,
        hashing_time=hashing_time,
        noise_type=noise_type,
        noise_stddev=noise_stddev,
        use_fec=use_fec,
        max_num_evals=max_num_evals,
        fec_remove_probability=fec_remove_probability,
        save_fec_history=save_fec_history,
        rng_seed=evaluator_rng_seed,
        test=True)
    evaluator2.fec = copy.deepcopy(evaluator.fec)

    max_time_budget = searcher.time_spent + hashing_time * 9
    searcher2 = evolution_search.EvolutionSearcher(
        nasbench=self._nasbench,
        evaluator=evaluator2,
        evolution_type=evolution_type,
        max_time_budget=max_time_budget,
        population_size=population_size,
        tournament_size=tournament_size,
        mutation_rate=mutation_rate,
        rng_seed=searcher_rng_seed,
        statistics=searcher.get_statistics())
    searcher2.time_spent = searcher.time_spent

    searcher2.run_search()

    statistics = searcher2.get_statistics()

    # From first pass of cache misses.
    expected_times = [
        0.0, 11157.675048828125, 12683.53114800882, 14434.180171290303,
        15881.224654305011, 17399.95196442609, 18950.05184148347,
        20502.776567202778
    ]
    # Add in FEC hash times for cache misses.
    num_cache_misses = len(expected_times) - 1
    for i in range(0, num_cache_misses + 1):
      expected_times[i] += hashing_time * i
    expected_best_valids = [
        0.0, 0.19378004670143126, 0.6863627939498267, 0.6863627939498267,
        0.6863627939498267, 0.7948589278789504, 0.7948589278789504,
        0.7948589278789504
    ]
    expected_best_tests = [
        0.0, 0.1932692289352417, 0.7232977219136525, 0.7232977219136525,
        0.7232977219136525, 0.8581393390434948, 0.8581393390434948,
        0.8581393390434948
    ]
    expected_num_cache_misses = [0, 1, 2, 3, 4, 5, 6, 7]
    expected_num_cache_partial_hits = [0, 0, 0, 0, 0, 0, 0, 0]
    expected_num_cache_full_hits = [0, 0, 0, 0, 0, 0, 0, 0]
    expected_cache_size = [0, 1, 2, 3, 4, 5, 6, 7]
    expected_model_hashes = [
        "", "2262111178723579678", "7468420669527941144", "1077607769737661496",
        "2075770728117857483", "8491132477128048640", "6510126918004834216",
        "5520427465221668864"
    ]

    # From second pass of cache hits.
    expected_times.extend([
        20512.776567202778, 20522.776567202778, 20532.776567202778,
        20542.776567202778, 20552.776567202778, 20562.776567202778,
        20572.776567202778, 20582.776567202778
    ])
    # Add in FEC hash times for cache misses.
    for i in range(num_cache_misses + 1, len(expected_times)):
      expected_times[i] += hashing_time * num_cache_misses
    expected_best_valids.extend([
        0.7948589278789504, 0.7948589278789504, 0.7948589278789504,
        0.7948589278789504, 0.7948589278789504, 0.7948589278789504,
        0.7948589278789504, 0.7948589278789504
    ])
    expected_best_tests.extend([
        0.8581393390434948, 0.8581393390434948, 0.8581393390434948,
        0.8581393390434948, 0.8581393390434948, 0.8581393390434948,
        0.8581393390434948, 0.8581393390434948
    ])
    expected_num_cache_misses.extend([7, 7, 7, 7, 7, 7, 7, 7])
    expected_num_cache_partial_hits.extend([0, 0, 0, 0, 0, 0, 0, 0])
    expected_num_cache_full_hits.extend([1, 2, 3, 4, 5, 6, 7, 8])
    expected_cache_size.extend([7, 7, 7, 7, 7, 7, 7, 7])
    expected_model_hashes.extend([
        "2262111178723579678", "7468420669527941144", "1077607769737661496",
        "2075770728117857483", "8491132477128048640", "6510126918004834216",
        "5520427465221668864", "8491132477128048640"
    ])

    # From second pass, a cache miss after the series of cache hits.
    num_cache_misses += 1
    expected_times.extend([22219.13045535685 + hashing_time * num_cache_misses])
    expected_best_valids.extend([0.8406047257580024])
    expected_best_tests.extend([0.3930430816854896])
    expected_num_cache_misses.extend([8])
    expected_num_cache_partial_hits.extend([0])
    expected_num_cache_full_hits.extend([8])
    expected_cache_size.extend([8])
    expected_model_hashes.extend(["784260650204103151"])

    # NOTE: These are different than the non-FEC version of this test above
    # because the FEC's RNG is out of sync from the Evaluator's RNG due to
    # different operations each does. They are identical when run_idx is
    # fixed as shown in the corresponding tests below.

    expected_statistics = {
        "times": expected_times,
        "best_valids": expected_best_valids,
        "best_tests": expected_best_tests,
        "num_cache_misses": expected_num_cache_misses,
        "num_cache_partial_hits": expected_num_cache_partial_hits,
        "num_cache_full_hits": expected_num_cache_full_hits,
        "cache_size": expected_cache_size,
        "model_hashes": expected_model_hashes
    }

    self.assertDictEqual(statistics, expected_statistics)

  def test_evolution_search_regularized_evolution_no_fim_no_fec_run_idx_0(self):
    rng = np.random.default_rng(RNG_SEED)
    evaluator_rng_seed = rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    evaluator_rng = np.random.default_rng(evaluator_rng_seed)
    _ = evaluator_rng.integers(  # evaluator_noise_generator_rng_seed
        low=1,
        high=constants.MAX_RNG_SEED,
        size=1)[0]
    hashing_time = 10.0
    evaluator = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        fixed_run_idx=0,
        mantissa_bits=24,
        hashing_time=hashing_time,
        noise_type="",
        noise_stddev=0.0,
        use_fec=False,
        max_num_evals=1,
        fec_remove_probability=0.0,
        rng_seed=evaluator_rng_seed,
        test=True)

    population_size = 5
    searcher_rng_seed = rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    searcher = evolution_search.EvolutionSearcher(
        nasbench=self._nasbench,
        evaluator=evaluator,
        evolution_type="regularized_evolution",
        max_time_budget=2e4,
        population_size=population_size,
        tournament_size=3,
        mutation_rate=1.0,
        rng_seed=searcher_rng_seed)

    searcher.run_search()

    statistics = searcher.get_statistics()

    expected_times = [
        0.0, 11155.85302734375, 12660.989821116968, 14173.20733597942,
        15620.251818994127, 17186.03347042582, 18736.1333474832,
        20177.402971399537
    ]
    expected_best_valids = [
        0.0, 0.19376001358032227, 0.32230874103327933, 0.4247075543893094,
        0.6481844482946302, 0.6965063927814837, 0.6965063927814837,
        0.6965063927814837
    ]
    expected_best_tests = [
        0.0, 0.19311898946762085, 0.9670391392747878, 0.7705209712704446,
        0.7288434725185676, 0.7025721170487348, 0.7025721170487348,
        0.7025721170487348
    ]

    expected_statistics = {
        "times": expected_times,
        "best_valids": expected_best_valids,
        "best_tests": expected_best_tests
    }

    self.assertDictEqual(statistics, expected_statistics)

  def test_evolution_search_regularized_evolution_no_fim_fec_run_idx_0(self):
    rng = np.random.default_rng(RNG_SEED)
    evaluator_rng_seed = rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    evaluator_rng = np.random.default_rng(evaluator_rng_seed)
    _ = evaluator_rng.integers(  # evaluator_noise_generator_rng_seed
        low=1,
        high=constants.MAX_RNG_SEED,
        size=1)[0]
    mantissa_bits = 24
    hashing_time = 10.0
    noise_type = ""
    noise_stddev = 0.0
    use_fec = True
    max_num_evals = 1
    fec_remove_probability = 0.0
    save_fec_history = True
    evaluator = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        fixed_run_idx=0,
        mantissa_bits=mantissa_bits,
        hashing_time=hashing_time,
        noise_type=noise_type,
        noise_stddev=noise_stddev,
        use_fec=use_fec,
        max_num_evals=max_num_evals,
        fec_remove_probability=fec_remove_probability,
        save_fec_history=save_fec_history,
        rng_seed=evaluator_rng_seed,
        test=True)

    searcher_rng_seed = rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    evolution_type = "regularized_evolution"
    max_time_budget = 2e4
    population_size = 5
    tournament_size = 3
    mutation_rate = 1.0
    searcher = evolution_search.EvolutionSearcher(
        nasbench=self._nasbench,
        evaluator=evaluator,
        evolution_type=evolution_type,
        max_time_budget=max_time_budget,
        population_size=population_size,
        tournament_size=tournament_size,
        mutation_rate=mutation_rate,
        rng_seed=searcher_rng_seed)

    searcher.run_search()

    # Force cache hits by reinitializing evaluator and copying FEC cache.
    evaluator2 = evaluator_lib.Evaluator(
        nasbench=self._nasbench,
        fixed_run_idx=0,
        mantissa_bits=mantissa_bits,
        hashing_time=hashing_time,
        noise_type=noise_type,
        noise_stddev=noise_stddev,
        use_fec=use_fec,
        max_num_evals=max_num_evals,
        fec_remove_probability=fec_remove_probability,
        save_fec_history=save_fec_history,
        rng_seed=evaluator_rng_seed,
        test=True)
    evaluator2.fec = copy.deepcopy(evaluator.fec)

    max_time_budget = searcher.time_spent + hashing_time * 9
    searcher2 = evolution_search.EvolutionSearcher(
        nasbench=self._nasbench,
        evaluator=evaluator2,
        evolution_type=evolution_type,
        max_time_budget=max_time_budget,
        population_size=population_size,
        tournament_size=tournament_size,
        mutation_rate=mutation_rate,
        rng_seed=searcher_rng_seed,
        statistics=searcher.get_statistics())
    searcher2.time_spent = searcher.time_spent

    searcher2.run_search()

    statistics = searcher2.get_statistics()

    # From first pass of cache misses.
    expected_times = [
        0.0, 11155.85302734375, 12660.989821116968, 14173.20733597942,
        15620.251818994127, 17186.03347042582, 18736.1333474832,
        20177.402971399537
    ]
    # Add in FEC hash times for cache misses.
    num_cache_misses = len(expected_times) - 1
    for i in range(0, num_cache_misses + 1):
      expected_times[i] += hashing_time * i

    expected_best_valids = [
        0.0, 0.19376001358032227, 0.32230874103327933, 0.4247075543893094,
        0.6481844482946302, 0.6965063927814837, 0.6965063927814837,
        0.6965063927814837
    ]
    expected_best_tests = [
        0.0, 0.19311898946762085, 0.9670391392747878, 0.7705209712704446,
        0.7288434725185676, 0.7025721170487348, 0.7025721170487348,
        0.7025721170487348
    ]
    expected_num_cache_misses = [0, 1, 2, 3, 4, 5, 6, 7]
    expected_num_cache_partial_hits = [0, 0, 0, 0, 0, 0, 0, 0]
    expected_num_cache_full_hits = [0, 0, 0, 0, 0, 0, 0, 0]
    expected_cache_size = [0, 1, 2, 3, 4, 5, 6, 7]
    expected_model_hashes = [
        "", "2262111178723579678", "7468420669527941144", "1077607769737661496",
        "2075770728117857483", "8491132477128048640", "6510126918004834216",
        "5520427465221668864"
    ]

    # From second pass of cache hits.
    expected_times.extend([
        20187.402971399537, 20197.402971399537, 20207.402971399537,
        20217.402971399537, 20227.402971399537, 20237.402971399537,
        20247.402971399537, 20257.402971399537
    ])
    # Add in FEC hash times for previous cache misses.
    for i in range(num_cache_misses + 1, len(expected_times)):
      expected_times[i] += hashing_time * num_cache_misses

    expected_best_valids.extend([
        0.6965063927814837, 0.6965063927814837, 0.6965063927814837,
        0.6965063927814837, 0.6965063927814837, 0.6965063927814837,
        0.6965063927814837, 0.6965063927814837
    ])
    expected_best_tests.extend([
        0.7025721170487348, 0.7025721170487348, 0.7025721170487348,
        0.7025721170487348, 0.7025721170487348, 0.7025721170487348,
        0.7025721170487348, 0.7025721170487348
    ])
    expected_num_cache_misses.extend([7, 7, 7, 7, 7, 7, 7, 7])
    expected_num_cache_partial_hits.extend([0, 0, 0, 0, 0, 0, 0, 0])
    expected_num_cache_full_hits.extend([1, 2, 3, 4, 5, 6, 7, 8])
    expected_cache_size.extend([7, 7, 7, 7, 7, 7, 7, 7])
    expected_model_hashes.extend([
        "2262111178723579678", "7468420669527941144", "1077607769737661496",
        "2075770728117857483", "8491132477128048640", "6510126918004834216",
        "5520427465221668864", "8491132477128048640"
    ])

    # From second pass, a cache miss after the series of cache hits.
    num_cache_misses += 1
    expected_times.extend([21893.75685955361 + hashing_time * num_cache_misses])
    expected_best_valids.extend([0.8406047257580024112])
    expected_best_tests.extend([0.3930430816854896])
    expected_num_cache_misses.extend([8])
    expected_num_cache_partial_hits.extend([0])
    expected_num_cache_full_hits.extend([8])
    expected_cache_size.extend([8])
    expected_model_hashes.extend(["784260650204103151"])

    expected_statistics = {
        "times": expected_times,
        "best_valids": expected_best_valids,
        "best_tests": expected_best_tests,
        "num_cache_misses": expected_num_cache_misses,
        "num_cache_partial_hits": expected_num_cache_partial_hits,
        "num_cache_full_hits": expected_num_cache_full_hits,
        "cache_size": expected_cache_size,
        "model_hashes": expected_model_hashes
    }

    self.assertDictEqual(statistics, expected_statistics)


if __name__ == "__main__":
  absltest.main()
