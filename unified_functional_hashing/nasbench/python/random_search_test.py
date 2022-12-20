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

"""Tests for random_search."""

import copy
from unittest import mock

from absl.testing import absltest
from unified_functional_hashing.nasbench.python import constants
from unified_functional_hashing.nasbench.python import evaluator as evaluator_lib
from unified_functional_hashing.nasbench.python import random_search
from unified_functional_hashing.nasbench.python import spec_validator_util
from unified_functional_hashing.nasbench.python import test_util
import numpy as np


RNG_SEED = 42


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

  def test_perform_search_iteration_no_fec(self):
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
    searcher = random_search.RandomSearcher(
        nasbench=self._nasbench,
        evaluator=evaluator,
        max_time_budget=max_time_budget,
        rng_seed=searcher_rng_seed)

    searcher.perform_search_iteration()
    statistics = searcher.get_statistics()

    # Calculate expected.
    expected_statistics = {
        "times": [0.0],
        "best_valids": [0.0],
        "best_tests": [0.0]
    }

    expected_rng = np.random.default_rng(RNG_SEED)
    expected_evaluator_rng = np.random.default_rng(
        expected_rng.integers(low=1, high=constants.MAX_RNG_SEED, size=1)[0])
    _ = expected_evaluator_rng.integers(  # expected_noise_generator_seed
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]

    test_util.update_expected_iterators(
        expected_evaluator_rng,
        expected_statistics["times"],
        expected_statistics["best_valids"],
        expected_statistics["best_tests"],
        graph_hash="e6a21c47208f1f9f3f954887665520a8",
        use_fec=use_fec,
        hashing_time=hashing_time)
    # expected_run_idx = 0
    # eval_outputs = 0.19376001358032227, 0.19311898946762085, 11155.85302734375
    # Since 0.19376001358032227 > 0.0, append these.

    self.assertDictEqual(statistics, expected_statistics)

  def test_perform_search_iteration_fec(self):
    rng = np.random.default_rng(RNG_SEED)
    evaluator_rng_seed = rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    searcher_rng_seed = rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    mantissa_bits = 24
    hashing_time = 10.0
    noise_type = ""
    noise_stddev = 0.0
    use_fec = True
    max_num_evals = 1
    fec_remove_probability = 0.0
    save_fec_history = True
    # First pass, cache miss.
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

    max_time_budget = 2e4
    searcher = random_search.RandomSearcher(
        nasbench=self._nasbench,
        evaluator=evaluator,
        max_time_budget=max_time_budget,
        rng_seed=searcher_rng_seed)

    # Iteration 0.
    searcher.perform_search_iteration()
    # Iteration 1.
    searcher.perform_search_iteration()

    # Second pass, allow a cache hit.
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

    max_time_budget = searcher.time_spent + hashing_time * 4
    searcher2 = random_search.RandomSearcher(
        nasbench=self._nasbench,
        evaluator=evaluator2,
        max_time_budget=max_time_budget,
        rng_seed=searcher_rng_seed,
        statistics=searcher.get_statistics())
    searcher2.time_spent = searcher.time_spent

    # Iteration 0.
    searcher2.perform_search_iteration()

    # Get statistics.
    statistics = searcher.get_statistics()

    # Calculate expected.
    expected_statistics = {
        "times": [0.0],
        "best_valids": [0.0],
        "best_tests": [0.0],
        "num_cache_misses": [0],
        "num_cache_partial_hits": [0],
        "num_cache_full_hits": [0],
        "cache_size": [0],
        "model_hashes": [""]
    }

    expected_rng = np.random.default_rng(RNG_SEED)
    expected_evaluator_rng = np.random.default_rng(
        expected_rng.integers(low=1, high=constants.MAX_RNG_SEED, size=1)[0])
    _ = expected_evaluator_rng.integers(  # expected_noise_generator_seed
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    expected_fec_rng_seed = expected_evaluator_rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]

    # First pass.
    expected_fec_rng = np.random.default_rng(expected_fec_rng_seed)
    # Iteration 0, cache miss.
    test_util.update_expected_iterators(
        expected_fec_rng,
        expected_statistics["times"],
        expected_statistics["best_valids"],
        expected_statistics["best_tests"],
        graph_hash="e6a21c47208f1f9f3f954887665520a8",
        use_fec=use_fec,
        cache_hit=False,
        hashing_time=hashing_time)
    # expected_run_idx = 0
    # eval_outputs = 0.19376001358032227, 0.19311898946762085, 11165.85302734375
    # Since 0.19376001358032227 > 0.0, append these.
    expected_statistics["num_cache_misses"].append(1)
    expected_statistics["num_cache_partial_hits"].append(0)
    expected_statistics["num_cache_full_hits"].append(0)
    expected_statistics["cache_size"].append(1)
    expected_statistics["model_hashes"].append("2262111178723579678")

    # Iteration 1, cache miss.
    test_util.update_expected_iterators(
        expected_fec_rng,
        expected_statistics["times"],
        expected_statistics["best_valids"],
        expected_statistics["best_tests"],
        graph_hash="256affa4b0191e323796f138db7b1164",
        use_fec=use_fec,
        cache_hit=False,
        hashing_time=hashing_time)
    # expected_run_idx = 1
    # eval_outputs = 0.6863627939498267, 0.7232977219136525, 1535.856099180696
    # Since 0.6863627939498267 > 0.19376001358032227, append these.
    expected_statistics["num_cache_misses"].append(2)
    expected_statistics["num_cache_partial_hits"].append(0)
    expected_statistics["num_cache_full_hits"].append(0)
    expected_statistics["cache_size"].append(2)
    expected_statistics["model_hashes"].append("7468420669527941144")

    # Second pass.
    expected_fec_rng2 = np.random.default_rng(expected_fec_rng_seed)
    # Iteration 0, cache hit.
    test_util.update_expected_iterators(
        expected_fec_rng2,
        expected_statistics["times"],
        expected_statistics["best_valids"],
        expected_statistics["best_tests"],
        graph_hash="e6a21c47208f1f9f3f954887665520a8",
        use_fec=use_fec,
        cache_hit=True,
        hashing_time=hashing_time)
    # expected_run_idx = 0
    # eval_outputs = 0.19376001358032227, 0.19311898946762085, 10.0
    # Since 0.19376001358032227 < 0.6863627939498267, DON'T append these.
    expected_statistics["num_cache_misses"].append(2)
    expected_statistics["num_cache_partial_hits"].append(0)
    expected_statistics["num_cache_full_hits"].append(1)
    expected_statistics["cache_size"].append(2)
    expected_statistics["model_hashes"].append("2262111178723579678")

    self.assertDictEqual(statistics, expected_statistics)

  def test_random_search_no_fec(self):
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
    searcher = random_search.RandomSearcher(
        nasbench=self._nasbench,
        evaluator=evaluator,
        max_time_budget=max_time_budget,
        rng_seed=searcher_rng_seed)

    searcher.run_search()

    # Calculate expected.
    expected_statistics = {
        "times": [0.0],
        "best_valids": [0.0],
        "best_tests": [0.0]
    }

    expected_rng = np.random.default_rng(RNG_SEED)
    expected_evaluator_rng = np.random.default_rng(
        expected_rng.integers(low=1, high=constants.MAX_RNG_SEED, size=1)[0])
    _ = expected_evaluator_rng.integers(  # expected_noise_generator_seed
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]

    # Iteration 0.
    test_util.update_expected_iterators(
        expected_evaluator_rng,
        expected_statistics["times"],
        expected_statistics["best_valids"],
        expected_statistics["best_tests"],
        graph_hash="e6a21c47208f1f9f3f954887665520a8",
        use_fec=use_fec,
        hashing_time=hashing_time)
    # expected_run_idx = 0
    # eval_outputs = 0.19376001358032227, 0.19311898946762085, 11155.85302734375
    # Since 0.19376001358032227 > 0.0, append these.

    # Iteration 1.
    test_util.update_expected_iterators(
        expected_evaluator_rng,
        expected_statistics["times"],
        expected_statistics["best_valids"],
        expected_statistics["best_tests"],
        graph_hash="256affa4b0191e323796f138db7b1164",
        use_fec=use_fec,
        hashing_time=hashing_time)
    # expected_run_idx = 1
    # eval_outputs = 0.6863627939498267, 0.7232977219136525, 1525.856099180696
    # Since 0.6863627939498267 > 0.19376001358032227, append these.

    # Iteration 2.
    test_util.update_expected_iterators(
        expected_evaluator_rng,
        expected_statistics["times"],
        expected_statistics["best_valids"],
        expected_statistics["best_tests"],
        graph_hash="0c50d88ea4244fdaaeba5953cb21dba5",
        use_fec=use_fec,
        hashing_time=hashing_time)
    # expected_run_idx = 0
    # eval_outputs = 0.4247075543893094, 0.7705209712704446, 1512.2175148624515
    # Since 0.4247075543893094 < 0.6863627939498267, DON'T append these.

    # Iteration 3.
    test_util.update_expected_iterators(
        expected_evaluator_rng,
        expected_statistics["times"],
        expected_statistics["best_valids"],
        expected_statistics["best_tests"],
        graph_hash="ae0401d99d86d0f838e80db250a19b39",
        use_fec=use_fec,
        hashing_time=hashing_time)
    # expected_run_idx = 0
    # eval_outputs = 0.6481844482946302, 0.7288434725185676, 1447.0444830147078
    # Since 0.6481844482946302 < 0.6863627939498267, DON'T append these.

    # Iteration 4.
    test_util.update_expected_iterators(
        expected_evaluator_rng,
        expected_statistics["times"],
        expected_statistics["best_valids"],
        expected_statistics["best_tests"],
        graph_hash="d0a5ff67e269a1ac4f305c618a88ac85",
        use_fec=use_fec,
        hashing_time=hashing_time)
    # expected_run_idx = 0
    # eval_outputs = 0.6965063927814837, 0.7025721170487348, 1565.7816514316921
    # Since 0.6965063927814837 > 0.6863627939498267, append these.

    # Iteration 5.
    test_util.update_expected_iterators(
        expected_evaluator_rng,
        expected_statistics["times"],
        expected_statistics["best_valids"],
        expected_statistics["best_tests"],
        graph_hash="95000ce507cc3a679842e61171a9c5be",
        use_fec=use_fec,
        hashing_time=hashing_time)
    # expected_run_idx = 0
    # eval_outputs = 0.6212108709784134, 0.8130059877791164, 1654.2448700112552
    # Since 0.6212108709784134 < 0.6965063927814837, DON'T append these.

    # Iteration 6.
    test_util.update_expected_iterators(
        expected_evaluator_rng,
        expected_statistics["times"],
        expected_statistics["best_valids"],
        expected_statistics["best_tests"],
        graph_hash="66ea4e36f994d4bf11102cbef10a3cb1",
        use_fec=use_fec,
        hashing_time=hashing_time)
    # expected_run_idx = 2
    # eval_outputs = 0.42259023668051743, 0.42228625522915175, 1686.9731762279062  # pylint: disable=line-too-long
    # Since 0.42259023668051743 < 0.6965063927814837, DON'T append these.

    statistics = searcher.get_statistics()

    self.assertDictEqual(statistics, expected_statistics)

  def test_random_search_fec(self):
    rng = np.random.default_rng(RNG_SEED)
    evaluator_rng_seed = rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    searcher_rng_seed = rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    mantissa_bits = 24
    hashing_time = 10.0
    noise_type = ""
    noise_stddev = 0.0
    use_fec = True
    max_num_evals = 1
    fec_remove_probability = 0.0
    save_fec_history = True
    # First pass, all cache misses.
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

    max_time_budget = 2e4
    searcher = random_search.RandomSearcher(
        nasbench=self._nasbench,
        evaluator=evaluator,
        max_time_budget=max_time_budget,
        rng_seed=searcher_rng_seed)

    searcher.run_search()

    # Second pass, allow some cache hits.
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

    max_time_budget = searcher.time_spent + hashing_time * 4
    searcher2 = random_search.RandomSearcher(
        nasbench=self._nasbench,
        evaluator=evaluator2,
        max_time_budget=max_time_budget,
        rng_seed=searcher_rng_seed,
        statistics=searcher.get_statistics())
    searcher2.time_spent = searcher.time_spent

    searcher2.run_search()

    # Get statistics.
    statistics = searcher2.get_statistics()

    # Calculate expected.
    expected_statistics = {
        "times": [0.0],
        "best_valids": [0.0],
        "best_tests": [0.0],
        "num_cache_misses": [0],
        "num_cache_partial_hits": [0],
        "num_cache_full_hits": [0],
        "cache_size": [0],
        "model_hashes": [""]
    }

    expected_rng = np.random.default_rng(RNG_SEED)
    expected_evaluator_rng = np.random.default_rng(
        expected_rng.integers(low=1, high=constants.MAX_RNG_SEED, size=1)[0])
    _ = expected_evaluator_rng.integers(  # expected_noise_generator_seed
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]
    expected_fec_rng_seed = expected_evaluator_rng.integers(
        low=1, high=constants.MAX_RNG_SEED, size=1)[0]

    # First pass.
    expected_fec_rng = np.random.default_rng(expected_fec_rng_seed)
    # Iteration 0, cache miss.
    test_util.update_expected_iterators(
        expected_fec_rng,
        expected_statistics["times"],
        expected_statistics["best_valids"],
        expected_statistics["best_tests"],
        graph_hash="e6a21c47208f1f9f3f954887665520a8",
        use_fec=use_fec,
        cache_hit=False,
        hashing_time=hashing_time)
    # expected_run_idx = 0
    # eval_outputs = 0.19376001358032227, 0.19311898946762085, 11165.85302734375
    # Since 0.19376001358032227 > 0.0, append these.
    expected_statistics["num_cache_misses"].append(1)
    expected_statistics["num_cache_partial_hits"].append(0)
    expected_statistics["num_cache_full_hits"].append(0)
    expected_statistics["cache_size"].append(1)
    expected_statistics["model_hashes"].append("2262111178723579678")

    # Iteration 1, cache miss.
    test_util.update_expected_iterators(
        expected_fec_rng,
        expected_statistics["times"],
        expected_statistics["best_valids"],
        expected_statistics["best_tests"],
        graph_hash="256affa4b0191e323796f138db7b1164",
        use_fec=use_fec,
        cache_hit=False,
        hashing_time=hashing_time)
    # expected_run_idx = 1
    # eval_outputs = 0.6863627939498267, 0.7232977219136525, 1535.856099180696
    # Since 0.6863627939498267 > 0.19376001358032227, append these.
    expected_statistics["num_cache_misses"].append(2)
    expected_statistics["num_cache_partial_hits"].append(0)
    expected_statistics["num_cache_full_hits"].append(0)
    expected_statistics["cache_size"].append(2)
    expected_statistics["model_hashes"].append("7468420669527941144")

    # Iteration 2, cache miss.
    test_util.update_expected_iterators(
        expected_fec_rng,
        expected_statistics["times"],
        expected_statistics["best_valids"],
        expected_statistics["best_tests"],
        graph_hash="0c50d88ea4244fdaaeba5953cb21dba5",
        use_fec=use_fec,
        cache_hit=False,
        hashing_time=hashing_time)
    # expected_run_idx = 0
    # eval_outputs = 0.4247075543893094, 0.7705209712704446, 1522.2175148624515
    # Since 0.4247075543893094 < 0.6863627939498267, DON'T append these.
    expected_statistics["num_cache_misses"].append(3)
    expected_statistics["num_cache_partial_hits"].append(0)
    expected_statistics["num_cache_full_hits"].append(0)
    expected_statistics["cache_size"].append(3)
    expected_statistics["model_hashes"].append("1077607769737661496")

    # Iteration 3, cache miss.
    test_util.update_expected_iterators(
        expected_fec_rng,
        expected_statistics["times"],
        expected_statistics["best_valids"],
        expected_statistics["best_tests"],
        graph_hash="ae0401d99d86d0f838e80db250a19b39",
        use_fec=use_fec,
        cache_hit=False,
        hashing_time=hashing_time)
    # expected_run_idx = 0
    # eval_outputs = 0.6481844482946302, 0.7288434725185676, 1457.0444830147078
    # Since 0.6481844482946302 < 0.6863627939498267, DON'T append these.
    expected_statistics["num_cache_misses"].append(4)
    expected_statistics["num_cache_partial_hits"].append(0)
    expected_statistics["num_cache_full_hits"].append(0)
    expected_statistics["cache_size"].append(4)
    expected_statistics["model_hashes"].append("2075770728117857483")

    # Iteration 4, cache miss.
    test_util.update_expected_iterators(
        expected_fec_rng,
        expected_statistics["times"],
        expected_statistics["best_valids"],
        expected_statistics["best_tests"],
        graph_hash="d0a5ff67e269a1ac4f305c618a88ac85",
        use_fec=use_fec,
        cache_hit=False,
        hashing_time=hashing_time)
    # expected_run_idx = 0
    # eval_outputs = 0.6965063927814837, 0.7025721170487348, 1575.7816514316921
    # Since 0.6965063927814837 > 0.6863627939498267, append these.
    expected_statistics["num_cache_misses"].append(5)
    expected_statistics["num_cache_partial_hits"].append(0)
    expected_statistics["num_cache_full_hits"].append(0)
    expected_statistics["cache_size"].append(5)
    expected_statistics["model_hashes"].append("8491132477128048640")

    # Iteration 5, cache miss.
    test_util.update_expected_iterators(
        expected_fec_rng,
        expected_statistics["times"],
        expected_statistics["best_valids"],
        expected_statistics["best_tests"],
        graph_hash="95000ce507cc3a679842e61171a9c5be",
        use_fec=use_fec,
        cache_hit=False,
        hashing_time=hashing_time)
    # expected_run_idx = 0
    # eval_outputs = 0.6212108709784134, 0.8130059877791164, 1664.2448700112552
    # Since 0.6212108709784134 < 0.6965063927814837, DON'T append these.
    expected_statistics["num_cache_misses"].append(6)
    expected_statistics["num_cache_partial_hits"].append(0)
    expected_statistics["num_cache_full_hits"].append(0)
    expected_statistics["cache_size"].append(6)
    expected_statistics["model_hashes"].append("1140311289072390563")

    # Iteration 6, cache miss.
    test_util.update_expected_iterators(
        expected_fec_rng,
        expected_statistics["times"],
        expected_statistics["best_valids"],
        expected_statistics["best_tests"],
        graph_hash="66ea4e36f994d4bf11102cbef10a3cb1",
        use_fec=use_fec,
        cache_hit=False,
        hashing_time=hashing_time)
    # expected_run_idx = 2
    # eval_outputs = 0.42259023668051743, 0.42228625522915175, 1696.9731762279062  # pylint: disable=line-too-long
    # Since 0.42259023668051743 < 0.6965063927814837, DON'T append these.
    expected_statistics["num_cache_misses"].append(7)
    expected_statistics["num_cache_partial_hits"].append(0)
    expected_statistics["num_cache_full_hits"].append(0)
    expected_statistics["cache_size"].append(7)
    expected_statistics["model_hashes"].append("6727226846357969280")

    # Second pass.
    expected_fec_rng2 = np.random.default_rng(expected_fec_rng_seed)
    # Iteration 0, cache hit.
    test_util.update_expected_iterators(
        expected_fec_rng2,
        expected_statistics["times"],
        expected_statistics["best_valids"],
        expected_statistics["best_tests"],
        graph_hash="e6a21c47208f1f9f3f954887665520a8",
        use_fec=use_fec,
        cache_hit=True,
        hashing_time=hashing_time)
    # expected_run_idx = 0
    # eval_outputs = 0.19376001358032227, 0.19311898946762085, 10.0
    # Since 0.19376001358032227 < 0.6965063927814837, DON'T append these.
    expected_statistics["num_cache_misses"].append(7)
    expected_statistics["num_cache_partial_hits"].append(0)
    expected_statistics["num_cache_full_hits"].append(1)
    expected_statistics["cache_size"].append(7)
    expected_statistics["model_hashes"].append("2262111178723579678")

    # Iteration 1, cache hit.
    test_util.update_expected_iterators(
        expected_fec_rng2,
        expected_statistics["times"],
        expected_statistics["best_valids"],
        expected_statistics["best_tests"],
        graph_hash="256affa4b0191e323796f138db7b1164",
        use_fec=use_fec,
        cache_hit=True,
        hashing_time=hashing_time)
    # expected_run_idx = 1
    # eval_outputs = 0.6863627939498267, 0.7232977219136525, 10.0
    # Since 0.6863627939498267 < 0.6965063927814837, DON'T append these.
    expected_statistics["num_cache_misses"].append(7)
    expected_statistics["num_cache_partial_hits"].append(0)
    expected_statistics["num_cache_full_hits"].append(2)
    expected_statistics["cache_size"].append(7)
    expected_statistics["model_hashes"].append("7468420669527941144")

    # Iteration 2, cache hit.
    test_util.update_expected_iterators(
        expected_fec_rng2,
        expected_statistics["times"],
        expected_statistics["best_valids"],
        expected_statistics["best_tests"],
        graph_hash="0c50d88ea4244fdaaeba5953cb21dba5",
        use_fec=use_fec,
        cache_hit=True,
        hashing_time=hashing_time)
    # expected_run_idx = 0
    # eval_outputs = 0.4247075543893094, 0.7705209712704446, 10.0
    # Since 0.4247075543893094 < 0.6965063927814837, DON'T append these.
    expected_statistics["num_cache_misses"].append(7)
    expected_statistics["num_cache_partial_hits"].append(0)
    expected_statistics["num_cache_full_hits"].append(3)
    expected_statistics["cache_size"].append(7)
    expected_statistics["model_hashes"].append("1077607769737661496")

    # Iteration 3, cache hit.
    test_util.update_expected_iterators(
        expected_fec_rng,
        expected_statistics["times"],
        expected_statistics["best_valids"],
        expected_statistics["best_tests"],
        graph_hash="ae0401d99d86d0f838e80db250a19b39",
        use_fec=use_fec,
        cache_hit=True,
        hashing_time=hashing_time)
    # expected_run_idx = 0
    # eval_outputs = 0.6481844482946302, 0.7288434725185676, 10.0
    # Since 0.6481844482946302 < 0.6965063927814837, DON'T append these.
    expected_statistics["num_cache_misses"].append(7)
    expected_statistics["num_cache_partial_hits"].append(0)
    expected_statistics["num_cache_full_hits"].append(4)
    expected_statistics["cache_size"].append(7)
    expected_statistics["model_hashes"].append("2075770728117857483")

    self.assertDictEqual(statistics, expected_statistics)


if __name__ == "__main__":
  absltest.main()
