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

"""Evolution searcher for NASBench.
"""

import heapq
from typing import Any, Dict, List, Optional, Tuple, Union

from unified_functional_hashing.nasbench.python import constants
from unified_functional_hashing.nasbench.python import evaluator as evaluator_lib
from unified_functional_hashing.nasbench.python import mutator as mutator_lib
from unified_functional_hashing.nasbench.python import random_spec_generator as random_spec_generator_lib
from unified_functional_hashing.nasbench.python import util
from nasbench import api
import numpy as np


class EvolutionSearcher():
  """EvolutionSearcher for finding best ModelSpecs through evolution.
  """

  def __init__(
      self,
      nasbench: api.NASBench,
      evaluator: evaluator_lib.Evaluator,
      evolution_type: str = "regularized_evolution",
      max_time_budget: float = 5e6,
      population_size: int = 50,
      save_child_history: bool = False,
      tournament_size: int = 10,
      mutation_rate: float = 1.0,
      rng_seed: Optional[int] = None,
      statistics: Optional[Dict[str, List[Union[float, int, str]]]] = None,
      population: Optional[List[Tuple[float, api.ModelSpec]]] = None):
    """Initializes EvolutionSearcher.

    Args:
      nasbench: NASBench instance.
      evaluator: Evaluator to calculate valid and test accuracies.
      evolution_type: Type of evolution: regularized_evolution or
        elitism_evolution.
      max_time_budget: Maximum time budget for search.
      population_size: The number of individuals in the population.
      save_child_history: Whether to save history of child. If True, its
        isomorphism-invariant graph hash, run indices, validation & test
        accuracies, and training times will be saved.
      tournament_size: The number of individuals selected for a tournament.
      mutation_rate: The probability to mutate a ModelSpec.
      rng_seed: Seed to initialize rng.
      statistics: Optional dict to track searcher's statistics.
      population: Optional list of fitness-ModelSpec tuples.
    """
    self.nasbench = nasbench
    self.evaluator = evaluator
    self.rng = np.random.default_rng(rng_seed)
    self.evolution_type = evolution_type
    self.max_time_budget = max_time_budget
    self.population_size = population_size
    self.save_child_history = save_child_history
    self.tournament_size = tournament_size
    self.random_spec_generator = random_spec_generator_lib.RandomSpecGenerator(
        nasbench,
        rng_seed=self.rng.integers(
            low=1, high=constants.MAX_RNG_SEED, size=1)[0])
    self.mutator = mutator_lib.Mutator(
        nasbench,
        mutation_rate,
        rng_seed=self.rng.integers(
            low=1, high=constants.MAX_RNG_SEED, size=1)[0])
    if population is None:
      self.population = []   # (validation, spec) tuples
    else:
      self.population = population
    self.time_spent = 0.0
    if statistics is None:
      self.statistics = self.initialze_statistics()
    else:
      self.statistics = statistics

  def initialze_statistics(self) -> Dict[str, List[Union[float, int, str]]]:
    """Initializes searcher's statistics.

    Returns:
      Dict of statistic lists.
    """
    statistics = {"times": [0.0], "best_valids": [0.0], "best_tests": [0.0]}
    if self.evaluator.has_fec() and self.evaluator.fec.save_history:
      statistics.update({
          "num_cache_misses": [0],
          "num_cache_partial_hits": [0],
          "num_cache_full_hits": [0],
          "cache_size": [0],
          # Model hash is computed by a Hasher using 4 accuracies from the
          # computed_stats of a ModelSpec
          "model_hashes": [""]})

    if self.save_child_history:
      statistics.update({
          "run_indices": [-1],
          "validation_accuracies": [0.0],
          "test_accuracies": [0.0],
          "training_times": [0.0],
          # Graph hash is the isomorphism-invariant graph hash of a ModelSpec.
          "graph_hashes": [""]})
    return statistics

  def get_statistics(self) -> Dict[str, List[Union[float, int, str]]]:
    """Gets searcher's statistics.

    Returns:
      Dict of statistic lists.
    """
    return self.statistics

  def get_population(self) -> List[Tuple[float, api.ModelSpec]]:
    """Gets searcher's population.

    Returns:
      List of 2-tuple fitness-ModelSpec pairs.
    """
    return self.population

  def add_random_individual_to_population(self) -> None:
    """Adds random individuals to population.
    """
    spec = self.random_spec_generator.random_spec()
    eval_outputs = self.evaluator.evaluate(spec)
    self.time_spent += eval_outputs[-1]
    self.statistics["times"].append(self.time_spent)

    util.update_best_fitnesses(
        (eval_outputs[1], eval_outputs[2]),
        self.statistics["best_valids"],
        self.statistics["best_tests"])

    if self.evaluator.has_fec() and self.evaluator.fec.save_history:
      fec_stats = self.evaluator.get_fec_statistics()
      for k, v in fec_stats.items():
        if k in self.statistics:
          self.statistics[k].append(v)

    if self.save_child_history:
      self.statistics["run_indices"].append(eval_outputs[0])
      self.statistics["validation_accuracies"].append(eval_outputs[1])
      self.statistics["test_accuracies"].append(eval_outputs[2])
      self.statistics["training_times"].append(eval_outputs[3])
      self.statistics["graph_hashes"].append(
          spec.hash_spec(
              canonical_ops=self.nasbench.config["available_ops"]))

    self.population.append((eval_outputs[1], spec))

  def seed_population(self) -> None:
    """Seeds population with random individuals.
    """
    for _ in range(self.population_size):
      self.add_random_individual_to_population()
      if self.time_spent >= self.max_time_budget:
        return

  def random_combination(
      self,
      iterable: List[Tuple[float, api.ModelSpec]],
      sample_size: int) -> Tuple[Tuple[float, api.ModelSpec], ...]:
    """Random selection from itertools.combinations(iterable, r).

    Args:
      iterable: list of fitness-ModelSpec tuples.
      sample_size: number of samples.

    Returns:
      Tuple of randomly sampled fitness-ModelSpec tuples.
    """
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(self.rng.choice(range(n), size=sample_size, replace=False))
    return tuple(pool[i] for i in indices)

  def get_best_spec_from_sample(
      self, sample: Tuple[Tuple[float, api.ModelSpec], ...]) -> api.ModelSpec:
    """Gets best spec from sample.

    Args:
      sample: Tuple of fitness-ModelSpec tuples.

    Returns:
      ModelSpec with highest fitness.
    """
    assert(sample), "Should have at least 1 sample!"
    best_fitness = sample[0][0]
    best_spec = sample[0][1]
    for (fitness, model_spec) in sample[1:]:
      if fitness >= best_fitness:
        best_fitness = fitness
        best_spec = model_spec
    return best_spec

  def get_best_spec_from_tournament(self) -> api.ModelSpec:
    """Gets best spec from tournament.

    Returns:
      ModelSpec with highest fitness.
    """
    sample = self.random_combination(self.population, self.tournament_size)
    return self.get_best_spec_from_sample(sample)

  def get_child_spec(self, parent_spec: api.ModelSpec) -> api.ModelSpec:
    """Gets a new ModelSpec from parent.

    Args:
      parent_spec: Parent ModelSpec.

    Returns:
      Child ModelSpec.
    """
    child_spec = self.mutator.mutate_spec(parent_spec)
    return child_spec

  def update_population(
      self,
      new_spec: api.ModelSpec,
      eval_outputs: Tuple[int, float, float, float],
      population_heap: Union[Any, List[Tuple[Any, int]]]) -> None:
    """Updates population with new individual.

    Args:
      new_spec: New ModelSpec to add to population.
      eval_outputs: 4-tuple of int run_idx and floats for valid and test
        accuracies and training time.
      population_heap: Min-heap of fitness-ModelSpec tuples.
    """
    if self.evolution_type == "regularized_evolution":
      # In regularized evolution, we kill the oldest individual in population.
      self.population.append((eval_outputs[1], new_spec))
      self.population.pop(0)
    elif self.evolution_type == "elitism_evolution":
      # In elitism evolution, we kill the worst individual in population. Here
      # we will just replace the worst with the new individual for same
      # effect.
      _, worst_idx = heapq.heappop(population_heap)
      heapq.heappush(population_heap, (eval_outputs[1], worst_idx))
      self.population[worst_idx] = (eval_outputs[1], new_spec)
    else:
      raise ValueError(
          "{} is not a valid evolution_type!".format(self.evolution_type))

  def perform_search_iteration(
      self, population_heap: Optional[Union[Any, List[Tuple[Any, int]]]] = None
      ) -> None:
    """Performs a single iteration of evolution search.

    Args:
      population_heap: Optional min-heap of fitness-ModelSpec tuples.
    """
    best_spec = self.get_best_spec_from_tournament()

    child_spec = self.get_child_spec(parent_spec=best_spec)

    eval_outputs = self.evaluator.evaluate(model_spec=child_spec)
    self.time_spent += eval_outputs[-1]
    self.statistics["times"].append(self.time_spent)

    util.update_best_fitnesses(
        (eval_outputs[1], eval_outputs[2]),
        self.statistics["best_valids"],
        self.statistics["best_tests"])

    if self.evaluator.has_fec() and self.evaluator.fec.save_history:
      fec_stats = self.evaluator.get_fec_statistics()
      for k, v in fec_stats.items():
        if k in self.statistics:
          self.statistics[k].append(v)

    if self.save_child_history:
      self.statistics["run_indices"].append(eval_outputs[0])
      self.statistics["validation_accuracies"].append(eval_outputs[1])
      self.statistics["test_accuracies"].append(eval_outputs[2])
      self.statistics["training_times"].append(eval_outputs[3])
      self.statistics["graph_hashes"].append(
          child_spec.hash_spec(
              canonical_ops=self.nasbench.config["available_ops"]))

    self.update_population(
        new_spec=child_spec,
        eval_outputs=eval_outputs,
        population_heap=population_heap)

  def run_search(self) -> None:
    """Run a single roll-out of evolution to a fixed time budget.
    """
    if not self.population:
      # For the first population_size individuals, seed the population with
      # randomly generated cells.
      self.seed_population()

    if self.time_spent >= self.max_time_budget:
      return

    population_heap = None
    if self.evolution_type == "elitism_evolution":
      # We can optimize search by using a min heap.
      population_heap = [
          (fitness, i) for i, (fitness, _) in enumerate(self.population)
      ]
      heapq.heapify(population_heap)

    # After the population is seeded, proceed with evolving the population.
    iterations = 0
    print("iterations = {}, time_spent = {}, best_valid = {}".format(
        iterations, self.time_spent, self.statistics["best_valids"][-1]))
    while self.time_spent < self.max_time_budget:
      self.perform_search_iteration(population_heap)

      iterations += 1
      if iterations % 100 == 0:
        print("iterations = {}, time_spent = {}, best_valid = {}".format(
            iterations, self.time_spent, self.statistics["best_valids"][-1]))
      if self.time_spent >= self.max_time_budget:
        print("iterations = {}, time_spent = {}, best_valid = {}".format(
            iterations, self.time_spent, self.statistics["best_valids"][-1]))
