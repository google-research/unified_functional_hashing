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

"""Tests for fec."""

from unittest import mock

from absl.testing import absltest
from unified_functional_hashing.nasbench.python import fec as fec_lib
from unified_functional_hashing.nasbench.python import hasher as hasher_lib
from unified_functional_hashing.nasbench.python import noise_generator as noise_generator_lib
from unified_functional_hashing.nasbench.python import test_util

RNG_SEED = 42


class FecTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    mock_nasbench = mock.MagicMock()
    mock_nasbench.get_metrics_from_spec = mock.MagicMock(
        side_effect=test_util.get_metrics)
    self._nasbench = mock_nasbench

    mock_model_spec = mock.MagicMock()
    mock_model_spec.graph_hash = "28cfc7874f6d200472e1a9dcd8650aa0"
    mock_model_spec.hash_spec = mock.MagicMock(
        return_value=mock_model_spec.graph_hash)
    self._model_spec = mock_model_spec

    mock_model_spec2 = mock.MagicMock()
    mock_model_spec2.graph_hash = "1"
    self._model_spec2 = mock_model_spec2

  def test_initial_empty_cache(self):
    noise_generator = noise_generator_lib.NoiseGenerator(
        noise_type="", noise_stddev=0.0, rng_seed=RNG_SEED)
    fec = fec_lib.FEC(
        nasbench=self._nasbench,
        noise_generator=noise_generator,
        mantissa_bits=24,
        hashing_time=10.0,
        max_num_evals=1,
        rng_seed=RNG_SEED)
    self.assertEmpty(fec.cache)

  def test_init_new_hash(self):
    noise_generator = noise_generator_lib.NoiseGenerator(
        noise_type="", noise_stddev=0.0, rng_seed=RNG_SEED)
    hashing_time = 10.0
    fec = fec_lib.FEC(
        nasbench=self._nasbench,
        noise_generator=noise_generator,
        mantissa_bits=24,
        hashing_time=hashing_time,
        max_num_evals=1,
        rng_seed=RNG_SEED)
    hasher = hasher_lib.Hasher(
        self._nasbench, mantissa_bits=24, hashing_time=hashing_time)
    model_hash, _ = hasher.get_unified_functional_hash(self._model_spec)
    fec.init_new_hash(model_hash=model_hash)
    expected_cache_1 = {
        model_hash: {
            "run_idx": -1,
            "valid_accuracy": 0.0,
            "test_accuracy": 0.0,
            "training_time": 0.0,
            "num_evals": 0
        }
    }
    self.assertDictEqual(fec.cache, expected_cache_1)
    self.assertEqual(fec.num_cache_misses, 0)
    self.assertEqual(fec.num_cache_partial_hits, 0)
    self.assertEqual(fec.num_cache_full_hits, 0)

    model_hash2, _ = hasher.get_unified_functional_hash(self._model_spec2)
    fec.init_new_hash(model_hash=model_hash2)
    expected_cache_2 = {
        model_hash: {
            "run_idx": -1,
            "valid_accuracy": 0.0,
            "test_accuracy": 0.0,
            "training_time": 0.0,
            "num_evals": 0
        },
        model_hash2: {
            "run_idx": -1,
            "valid_accuracy": 0.0,
            "test_accuracy": 0.0,
            "training_time": 0.0,
            "num_evals": 0
        }
    }
    self.assertDictEqual(fec.cache, expected_cache_2)
    self.assertEqual(fec.num_cache_misses, 0)
    self.assertEqual(fec.num_cache_partial_hits, 0)
    self.assertEqual(fec.num_cache_full_hits, 0)

  def test_update_cache(self):
    noise_generator = noise_generator_lib.NoiseGenerator(
        noise_type="", noise_stddev=0.0, rng_seed=RNG_SEED)
    hashing_time = 10.0
    fec = fec_lib.FEC(
        nasbench=self._nasbench,
        noise_generator=noise_generator,
        mantissa_bits=24,
        hashing_time=hashing_time,
        max_num_evals=1,
        rng_seed=RNG_SEED)
    self._model_spec.graph_hash = "28cfc7874f6d200472e1a9dcd8650aa0"
    hasher = hasher_lib.Hasher(
        self._nasbench, mantissa_bits=24, hashing_time=hashing_time)
    model_hash, _ = hasher.get_unified_functional_hash(self._model_spec)
    fec.init_new_hash(model_hash=model_hash)
    fec.update_cache(model_hash=model_hash, model_spec=self._model_spec)
    # From computed_stats[108][0].
    expected_cache = {
        model_hash: {
            "run_idx": 0,
            "valid_accuracy": 0.9376001358032227,
            "test_accuracy": 0.9311898946762085,
            "training_time": 1155.85302734375,
            "num_evals": 1
        }
    }
    self.assertDictEqual(fec.cache, expected_cache)
    # Cache miss is still 0 since get_eval_outputs wasn't called.
    self.assertEqual(fec.num_cache_misses, 0)
    self.assertEqual(fec.num_cache_partial_hits, 0)
    self.assertEqual(fec.num_cache_full_hits, 0)

  def test_get_eval_outputs_cache_miss(self):
    noise_generator = noise_generator_lib.NoiseGenerator(
        noise_type="", noise_stddev=0.0, rng_seed=RNG_SEED)
    hashing_time = 10.0
    fec = fec_lib.FEC(
        nasbench=self._nasbench,
        noise_generator=noise_generator,
        mantissa_bits=24,
        hashing_time=10.0,
        max_num_evals=1,
        rng_seed=RNG_SEED)
    self._model_spec.graph_hash = "28cfc7874f6d200472e1a9dcd8650aa0"
    hasher = hasher_lib.Hasher(nasbench=self._nasbench, mantissa_bits=24)
    model_hash, _ = hasher.get_unified_functional_hash(self._model_spec)
    eval_outputs = fec.get_eval_outputs(model_spec=self._model_spec)
    # From computed_stats[108][0].
    run_idx, valid_acc, test_acc, training_time = eval_outputs
    self.assertEqual(run_idx, 0)
    self.assertEqual(valid_acc, 0.9376001358032227)
    self.assertEqual(test_acc, 0.9311898946762085)
    self.assertEqual(training_time, 1155.85302734375 + hashing_time)

    expected_cache = {
        model_hash: {
            "run_idx": 0,
            "valid_accuracy": 0.9376001358032227,
            "test_accuracy": 0.9311898946762085,
            "training_time": 1155.85302734375,
            "num_evals": 1
        }
    }
    self.assertDictEqual(fec.cache, expected_cache)
    self.assertEqual(fec.num_cache_misses, 1)
    self.assertEqual(fec.num_cache_partial_hits, 0)
    self.assertEqual(fec.num_cache_full_hits, 0)

  def test_get_eval_outputs_cache_hit(self):
    noise_generator = noise_generator_lib.NoiseGenerator(
        noise_type="", noise_stddev=0.0, rng_seed=RNG_SEED)
    hashing_time = 10.0
    fec = fec_lib.FEC(
        nasbench=self._nasbench,
        noise_generator=noise_generator,
        mantissa_bits=24,
        hashing_time=10.0,
        max_num_evals=1,
        rng_seed=RNG_SEED)
    self._model_spec.graph_hash = "28cfc7874f6d200472e1a9dcd8650aa0"
    hasher = hasher_lib.Hasher(
        nasbench=self._nasbench, mantissa_bits=24, hashing_time=hashing_time)
    model_hash, _ = hasher.get_unified_functional_hash(self._model_spec)
    eval_outputs = fec.get_eval_outputs(model_spec=self._model_spec)
    self.assertEqual(fec.num_cache_misses, 1)
    self.assertEqual(fec.num_cache_partial_hits, 0)
    self.assertEqual(fec.num_cache_full_hits, 0)
    eval_outputs = fec.get_eval_outputs(model_spec=self._model_spec)
    # From computed_stats[108][0].
    run_idx, valid_acc, test_acc, training_time = eval_outputs
    self.assertEqual(run_idx, 0)
    self.assertEqual(valid_acc, 0.9376001358032227)
    self.assertEqual(test_acc, 0.9311898946762085)
    self.assertEqual(training_time, hashing_time)

    expected_cache = {
        model_hash: {
            "run_idx": 0,
            "valid_accuracy": 0.9376001358032227,
            "test_accuracy": 0.9311898946762085,
            "training_time": 1155.85302734375,
            "num_evals": 1
        }
    }
    self.assertDictEqual(fec.cache, expected_cache)
    self.assertEqual(fec.num_cache_misses, 1)
    self.assertEqual(fec.num_cache_partial_hits, 0)
    self.assertEqual(fec.num_cache_full_hits, 1)

  def test_get_eval_outputs_standard_aggregation(self):
    noise_generator = noise_generator_lib.NoiseGenerator(
        noise_type="", noise_stddev=0.0, rng_seed=RNG_SEED)
    hashing_time = 10.0
    fec = fec_lib.FEC(
        nasbench=self._nasbench,
        noise_generator=noise_generator,
        mantissa_bits=24,
        hashing_time=hashing_time,
        max_num_evals=2,
        rng_seed=RNG_SEED)
    self._model_spec.graph_hash = "28cfc7874f6d200472e1a9dcd8650aa0"
    hasher = hasher_lib.Hasher(
        nasbench=self._nasbench, mantissa_bits=24, hashing_time=hashing_time)
    model_hash, _ = hasher.get_unified_functional_hash(self._model_spec)
    # Cache miss.
    eval_outputs = fec.get_eval_outputs(model_spec=self._model_spec)
    # From computed_stats[108][0].
    run_idx, valid_acc, test_acc, training_time = eval_outputs
    self.assertEqual(run_idx, 0)
    self.assertEqual(valid_acc, 0.9376001358032227)
    self.assertEqual(test_acc, 0.9311898946762085)
    self.assertEqual(training_time, 1155.85302734375 + hashing_time)
    self.assertEqual(fec.cache[model_hash]["num_evals"], 1)
    self.assertEqual(fec.num_cache_misses, 1)
    self.assertEqual(fec.num_cache_partial_hits, 0)
    self.assertEqual(fec.num_cache_full_hits, 0)

    # Partial cache hit.
    eval_outputs = fec.get_eval_outputs(model_spec=self._model_spec)
    # mean(computed_stats[108][0], computed_stats[108][2]).
    run_idx, valid_acc, test_acc, training_time = eval_outputs
    self.assertEqual(run_idx, 2)
    # mean([0.9376001358032227, 0.9336938858032227])
    self.assertAlmostEqual(valid_acc, 0.935647010803223, delta=1e-8)
    # mean([0.9311898946762085, 0.9286859035491943])
    self.assertAlmostEqual(test_acc, 0.929937899112701, delta=1e-8)
    # mean([1155.85302734375, 1154.361083984375])
    self.assertAlmostEqual(
        training_time, 1155.1070556640625 + hashing_time, delta=1e-8)
    self.assertEqual(fec.cache[model_hash]["num_evals"], 2)
    self.assertEqual(fec.num_cache_misses, 1)
    self.assertEqual(fec.num_cache_partial_hits, 1)
    self.assertEqual(fec.num_cache_full_hits, 0)

    # Full cache hit.
    eval_outputs = fec.get_eval_outputs(model_spec=self._model_spec)
    # mean(computed_stats[108][0], computed_stats[108][2]).
    run_idx, valid_acc, test_acc, training_time = eval_outputs
    self.assertEqual(run_idx, 2)
    # mean([0.9376001358032227, 0.9336938858032227])
    self.assertAlmostEqual(valid_acc, 0.935647010803223, delta=1e-8)
    # mean([0.9311898946762085, 0.9286859035491943])
    self.assertAlmostEqual(test_acc, 0.929937899112701, delta=1e-8)
    self.assertEqual(training_time, hashing_time)
    self.assertEqual(fec.cache[model_hash]["num_evals"], 2)
    self.assertEqual(fec.num_cache_misses, 1)
    self.assertEqual(fec.num_cache_partial_hits, 1)
    self.assertEqual(fec.num_cache_full_hits, 1)

  def test_get_eval_outputs_standard_aggregation_large_max_num_evals(self):
    noise_generator = noise_generator_lib.NoiseGenerator(
        noise_type="", noise_stddev=0.0, rng_seed=RNG_SEED)
    max_num_evals = 1000
    hashing_time = 10.0
    fec = fec_lib.FEC(
        nasbench=self._nasbench,
        noise_generator=noise_generator,
        mantissa_bits=24,
        hashing_time=hashing_time,
        max_num_evals=max_num_evals,
        rng_seed=RNG_SEED)
    self._model_spec.graph_hash = "28cfc7874f6d200472e1a9dcd8650aa0"
    hasher = hasher_lib.Hasher(
        nasbench=self._nasbench, mantissa_bits=24, hashing_time=hashing_time)
    model_hash, _ = hasher.get_unified_functional_hash(self._model_spec)
    # Cache miss.
    _ = fec.get_eval_outputs(model_spec=self._model_spec)
    self.assertEqual(fec.cache[model_hash]["num_evals"], 1)
    self.assertEqual(fec.num_cache_misses, 1)
    self.assertEqual(fec.num_cache_partial_hits, 0)
    self.assertEqual(fec.num_cache_full_hits, 0)
    for _ in range(max_num_evals - 2):
      # Partial cache hit.
      _ = fec.get_eval_outputs(model_spec=self._model_spec)
    self.assertEqual(fec.cache[model_hash]["num_evals"], max_num_evals - 1)
    self.assertEqual(fec.num_cache_misses, 1)
    self.assertEqual(fec.num_cache_partial_hits, max_num_evals - 2)
    self.assertEqual(fec.num_cache_full_hits, 0)
    # Partial cache hit.
    (_, _, _, training_time) = fec.get_eval_outputs(model_spec=self._model_spec)
    self.assertEqual(fec.cache[model_hash]["num_evals"], max_num_evals)
    self.assertGreater(training_time, hashing_time)
    self.assertEqual(fec.num_cache_misses, 1)
    self.assertEqual(fec.num_cache_partial_hits, max_num_evals - 1)
    self.assertEqual(fec.num_cache_full_hits, 0)
    # Full cache hit.
    (_, _, _, training_time) = fec.get_eval_outputs(model_spec=self._model_spec)
    self.assertEqual(fec.cache[model_hash]["num_evals"], max_num_evals)
    self.assertEqual(training_time, hashing_time)
    self.assertEqual(fec.num_cache_misses, 1)
    self.assertEqual(fec.num_cache_partial_hits, max_num_evals - 1)
    self.assertEqual(fec.num_cache_full_hits, 1)

  def test_retrieve_correct_hashs_eval_outputs(self):
    noise_generator = noise_generator_lib.NoiseGenerator(
        noise_type="", noise_stddev=0.0, rng_seed=RNG_SEED)
    hashing_time = 10.0
    fec = fec_lib.FEC(
        nasbench=self._nasbench,
        noise_generator=noise_generator,
        mantissa_bits=24,
        hashing_time=hashing_time,
        max_num_evals=1,
        rng_seed=RNG_SEED)
    # Cache miss.
    eval_outputs = fec.get_eval_outputs(model_spec=self._model_spec)
    # From computed_stats[108][0].
    run_idx, valid_acc, test_acc, training_time = eval_outputs
    self.assertEqual(run_idx, 0)
    self.assertEqual(valid_acc, 0.9376001358032227)
    self.assertEqual(test_acc, 0.9311898946762085)
    self.assertEqual(training_time, 1155.85302734375 + hashing_time)

    hasher = hasher_lib.Hasher(
        self._nasbench, mantissa_bits=24, hashing_time=hashing_time)
    model_hash, _ = hasher.get_unified_functional_hash(self._model_spec)
    expected_cache = {
        model_hash: {
            "run_idx": 0,
            "valid_accuracy": 0.9376001358032227,
            "test_accuracy": 0.9311898946762085,
            "training_time": 1155.85302734375,
            "num_evals": 1
        }
    }
    self.assertDictEqual(fec.cache, expected_cache)
    self.assertEqual(fec.num_cache_misses, 1)
    self.assertEqual(fec.num_cache_partial_hits, 0)
    self.assertEqual(fec.num_cache_full_hits, 0)

    # Cache miss.
    model_hash2, _ = hasher.get_unified_functional_hash(self._model_spec2)
    fec.init_new_hash(model_hash=model_hash2)
    fec.update_cache(model_hash=model_hash2, model_spec=self._model_spec2)
    # From computed_stats[108][2].
    expected_cache = {
        model_hash: {
            "run_idx": 0,
            "valid_accuracy": 0.9376001358032227,
            "test_accuracy": 0.9311898946762085,
            "training_time": 1155.85302734375,
            "num_evals": 1
        },
        model_hash2: {
            "run_idx": 2,
            "valid_accuracy": 0.00124108,
            "test_accuracy": 0.00121108,
            "training_time": 0.00123108,
            "num_evals": 1
        }
    }
    self.assertDictEqual(fec.cache, expected_cache)

    # Cache hit for first hash.
    eval_outputs = fec.get_eval_outputs(model_spec=self._model_spec)
    # From computed_stats[108][0].
    run_idx, valid_acc, test_acc, training_time = eval_outputs
    self.assertEqual(run_idx, 0)
    self.assertEqual(valid_acc, 0.9376001358032227)
    self.assertEqual(test_acc, 0.9311898946762085)
    self.assertEqual(training_time, hashing_time)
    self.assertEqual(fec.num_cache_misses, 1)
    self.assertEqual(fec.num_cache_partial_hits, 0)
    self.assertEqual(fec.num_cache_full_hits, 1)

  def test_remove_prob_0_forgets_nothing(self):
    """Tests when remove_probability = 0, nothing should be forgotten."""
    noise_generator = noise_generator_lib.NoiseGenerator(
        noise_type="", noise_stddev=0.0, rng_seed=RNG_SEED)
    hashing_time = 10.0
    fec = fec_lib.FEC(
        nasbench=self._nasbench,
        noise_generator=noise_generator,
        mantissa_bits=24,
        hashing_time=hashing_time,
        max_num_evals=1,
        remove_probability=0.0,
        rng_seed=RNG_SEED)
    # Three back-to-back cache misses.
    hasher = hasher_lib.Hasher(
        self._nasbench, mantissa_bits=24, hashing_time=hashing_time)
    model_hashes = []
    for i in range(1, 4):
      self._model_spec.graph_hash = str(i)
      model_hash, _ = hasher.get_unified_functional_hash(self._model_spec)
      fec.init_new_hash(model_hash=model_hash)
      fec.update_cache(model_hash=model_hash, model_spec=self._model_spec)
      model_hashes.append(model_hash)
    expected_cache = {
        model_hashes[0]: {
            "run_idx": 0,
            "valid_accuracy": 0.00104108,
            "test_accuracy": 0.00101108,
            "training_time": 0.00103108,
            "num_evals": 1
        },
        model_hashes[1]: {
            "run_idx": 2,
            "valid_accuracy": 0.00224108,
            "test_accuracy": 0.00221108,
            "training_time": 0.00223108,
            "num_evals": 1
        },
        model_hashes[2]: {
            "run_idx": 1,
            "valid_accuracy": 0.00314108,
            "test_accuracy": 0.00311108,
            "training_time": 0.00313108,
            "num_evals": 1
        }
    }
    self.assertDictEqual(fec.cache, expected_cache)

    # Cache miss for new entry.
    self._model_spec.graph_hash = "28cfc7874f6d200472e1a9dcd8650aa0"
    eval_outputs = fec.get_eval_outputs(model_spec=self._model_spec)
    # From computed_stats[108][1].
    run_idx, valid_acc, test_acc, training_time = eval_outputs
    # From newest entry.
    self.assertEqual(run_idx, 1)
    self.assertEqual(valid_acc, 0.9378004670143127)
    self.assertEqual(test_acc, 0.932692289352417)
    self.assertEqual(training_time, 1157.675048828125 + hashing_time)
    # Newest entry added to cache with existing remaining.
    model_hash, _ = hasher.get_unified_functional_hash(self._model_spec)
    expected_cache[model_hash] = {
        "run_idx": 1,
        "valid_accuracy": 0.9378004670143127,
        "test_accuracy": 0.932692289352417,
        "training_time": 1157.675048828125,
        "num_evals": 1
    }
    self.assertDictEqual(fec.cache, expected_cache)
    self.assertEqual(fec.num_cache_misses, 1)
    self.assertEqual(fec.num_cache_partial_hits, 0)
    self.assertEqual(fec.num_cache_full_hits, 0)

    # Cache hit for new entry.
    eval_outputs = fec.get_eval_outputs(model_spec=self._model_spec)
    # From computed_stats[108][1].
    run_idx, valid_acc, test_acc, training_time = eval_outputs
    # From newest entry.
    self.assertEqual(run_idx, 1)
    self.assertEqual(valid_acc, 0.9378004670143127)
    self.assertEqual(test_acc, 0.932692289352417)
    self.assertEqual(training_time, hashing_time)
    # Nothing was forgotten.
    self.assertDictEqual(fec.cache, expected_cache)
    self.assertEqual(fec.num_cache_misses, 1)
    self.assertEqual(fec.num_cache_partial_hits, 0)
    self.assertEqual(fec.num_cache_full_hits, 1)

  def test_remove_prob_100_forgets_correctly(self):
    """Tests when remove_probability = 1, only current hash should be forgotten.
    """
    noise_generator = noise_generator_lib.NoiseGenerator(
        noise_type="", noise_stddev=0.0, rng_seed=RNG_SEED)
    hashing_time = 10.0
    fec = fec_lib.FEC(
        nasbench=self._nasbench,
        noise_generator=noise_generator,
        mantissa_bits=24,
        hashing_time=hashing_time,
        max_num_evals=1,
        remove_probability=1.0,
        rng_seed=RNG_SEED)
    # Three back-to-back cache misses.
    hasher = hasher_lib.Hasher(
        self._nasbench, mantissa_bits=24, hashing_time=hashing_time)
    model_hashes = []
    for i in range(1, 4):
      self._model_spec.graph_hash = str(i)
      model_hash, _ = hasher.get_unified_functional_hash(self._model_spec)
      fec.init_new_hash(model_hash=model_hash)
      fec.update_cache(model_hash=model_hash, model_spec=self._model_spec)
      model_hashes.append(model_hash)
    expected_cache = {
        model_hashes[0]: {
            "run_idx": 0,
            "valid_accuracy": 0.00104108,
            "test_accuracy": 0.00101108,
            "training_time": 0.00103108,
            "num_evals": 1
        },
        model_hashes[1]: {
            "run_idx": 2,
            "valid_accuracy": 0.00224108,
            "test_accuracy": 0.00221108,
            "training_time": 0.00223108,
            "num_evals": 1
        },
        model_hashes[2]: {
            "run_idx": 1,
            "valid_accuracy": 0.00314108,
            "test_accuracy": 0.00311108,
            "training_time": 0.00313108,
            "num_evals": 1
        }
    }
    self.assertDictEqual(fec.cache, expected_cache)

    # Cache miss for new entry.
    self._model_spec.graph_hash = "28cfc7874f6d200472e1a9dcd8650aa0"
    eval_outputs = fec.get_eval_outputs(model_spec=self._model_spec)
    # From computed_stats[108][1].
    run_idx, valid_acc, test_acc, training_time = eval_outputs
    # From newest entry.
    self.assertEqual(run_idx, 1)
    self.assertEqual(valid_acc, 0.9378004670143127)
    self.assertEqual(test_acc, 0.932692289352417)
    self.assertEqual(training_time, 1157.675048828125 + hashing_time)
    # Newest entry added to cache with existing remaining.
    model_hash, _ = hasher.get_unified_functional_hash(self._model_spec)
    expected_cache[model_hash] = {
        "run_idx": 1,
        "valid_accuracy": 0.9378004670143127,
        "test_accuracy": 0.932692289352417,
        "training_time": 1157.675048828125,
        "num_evals": 1
    }
    self.assertDictEqual(fec.cache, expected_cache)
    self.assertEqual(fec.num_cache_misses, 1)
    self.assertEqual(fec.num_cache_partial_hits, 0)
    self.assertEqual(fec.num_cache_full_hits, 0)

    # Cache hit for new entry.
    eval_outputs = fec.get_eval_outputs(model_spec=self._model_spec)
    # From computed_stats[108][1].
    run_idx, valid_acc, test_acc, training_time = eval_outputs
    # From newest entry.
    self.assertEqual(run_idx, 1)
    self.assertEqual(valid_acc, 0.9378004670143127)
    self.assertEqual(test_acc, 0.932692289352417)
    self.assertEqual(training_time, hashing_time)
    # New entry was forgotten, but nothing else.
    expected_cache.pop(model_hash)
    self.assertDictEqual(fec.cache, expected_cache)
    self.assertEqual(fec.num_cache_misses, 1)
    self.assertEqual(fec.num_cache_partial_hits, 0)
    self.assertEqual(fec.num_cache_full_hits, 1)

    # Add removed hash back to cache to remove it again!
    # Cache miss for new entry.
    eval_outputs = fec.get_eval_outputs(model_spec=self._model_spec)
    # From computed_stats[108][0].
    run_idx, valid_acc, test_acc, training_time = eval_outputs
    # From newest entry.
    self.assertEqual(run_idx, 0)
    self.assertEqual(valid_acc, 0.9376001358032227)
    self.assertEqual(test_acc, 0.9311898946762085)
    self.assertEqual(training_time, 1155.85302734375 + hashing_time)
    # Newest entry added to cache with existing remaining.
    expected_cache[model_hash] = {
        "run_idx": 0,
        "valid_accuracy": 0.9376001358032227,
        "test_accuracy": 0.9311898946762085,
        "training_time": 1155.85302734375,
        "num_evals": 1
    }
    self.assertDictEqual(fec.cache, expected_cache)
    self.assertEqual(fec.num_cache_misses, 2)
    self.assertEqual(fec.num_cache_partial_hits, 0)
    self.assertEqual(fec.num_cache_full_hits, 1)

    # Cache hit for new entry.
    eval_outputs = fec.get_eval_outputs(model_spec=self._model_spec)
    # From computed_stats[108][1].
    run_idx, valid_acc, test_acc, training_time = eval_outputs
    # From newest entry.
    self.assertEqual(run_idx, 0)  # from previous evaluation when added to cache
    self.assertEqual(valid_acc, 0.9376001358032227)
    self.assertEqual(test_acc, 0.9311898946762085)
    self.assertEqual(training_time, hashing_time)
    # New entry was forgotten, but nothing else.
    expected_cache.pop(model_hash)
    self.assertDictEqual(fec.cache, expected_cache)
    self.assertEqual(fec.num_cache_misses, 2)
    self.assertEqual(fec.num_cache_partial_hits, 0)
    self.assertEqual(fec.num_cache_full_hits, 2)

  def test_get_cache_statistics_empty(self):
    noise_generator = noise_generator_lib.NoiseGenerator(
        noise_type="", noise_stddev=0.0, rng_seed=RNG_SEED)
    fec = fec_lib.FEC(
        nasbench=self._nasbench,
        noise_generator=noise_generator,
        mantissa_bits=24,
        hashing_time=10.0,
        max_num_evals=1,
        remove_probability=0.0,
        rng_seed=RNG_SEED)

    cache_stats = fec.get_cache_statistics()
    self.assertEqual(cache_stats["num_cache_misses"], 0)
    self.assertEqual(cache_stats["num_cache_partial_hits"], 0)
    self.assertEqual(cache_stats["num_cache_full_hits"], 0)
    self.assertEqual(cache_stats["cache_size"], 0)

  def test_get_cache_statistics_with_history_empty(self):
    noise_generator = noise_generator_lib.NoiseGenerator(
        noise_type="", noise_stddev=0.0, rng_seed=RNG_SEED)
    fec = fec_lib.FEC(
        nasbench=self._nasbench,
        noise_generator=noise_generator,
        mantissa_bits=24,
        hashing_time=10.0,
        max_num_evals=1,
        remove_probability=0.0,
        save_history=True,
        rng_seed=RNG_SEED)

    cache_stats = fec.get_cache_statistics()
    self.assertEqual(cache_stats["num_cache_misses"], 0)
    self.assertEqual(cache_stats["num_cache_partial_hits"], 0)
    self.assertEqual(cache_stats["num_cache_full_hits"], 0)
    self.assertEqual(cache_stats["cache_size"], 0)
    self.assertEqual(cache_stats["model_hashes"], "")

  def test_get_cache_statistics_cache_miss(self):
    noise_generator = noise_generator_lib.NoiseGenerator(
        noise_type="", noise_stddev=0.0, rng_seed=RNG_SEED)
    fec = fec_lib.FEC(
        nasbench=self._nasbench,
        noise_generator=noise_generator,
        mantissa_bits=24,
        hashing_time=10.0,
        max_num_evals=1,
        remove_probability=0.0,
        rng_seed=RNG_SEED)

    self._model_spec.graph_hash = "28cfc7874f6d200472e1a9dcd8650aa0"
    _ = fec.get_eval_outputs(model_spec=self._model_spec)

    cache_stats = fec.get_cache_statistics()
    self.assertEqual(cache_stats["num_cache_misses"], 1)
    self.assertEqual(cache_stats["num_cache_partial_hits"], 0)
    self.assertEqual(cache_stats["num_cache_full_hits"], 0)
    self.assertEqual(cache_stats["cache_size"], 1)

  def test_get_cache_statistics_cache_hit(self):
    noise_generator = noise_generator_lib.NoiseGenerator(
        noise_type="", noise_stddev=0.0, rng_seed=RNG_SEED)
    fec = fec_lib.FEC(
        nasbench=self._nasbench,
        noise_generator=noise_generator,
        mantissa_bits=24,
        hashing_time=10.0,
        max_num_evals=1,
        remove_probability=0.0,
        rng_seed=RNG_SEED)

    self._model_spec.graph_hash = "28cfc7874f6d200472e1a9dcd8650aa0"
    _ = fec.get_eval_outputs(model_spec=self._model_spec)
    _ = fec.get_eval_outputs(model_spec=self._model_spec)

    cache_stats = fec.get_cache_statistics()
    self.assertEqual(cache_stats["num_cache_misses"], 1)
    self.assertEqual(cache_stats["num_cache_partial_hits"], 0)
    self.assertEqual(cache_stats["num_cache_full_hits"], 1)
    self.assertEqual(cache_stats["cache_size"], 1)

  def test_get_cache_statistics_cache_partial_hit(self):
    noise_generator = noise_generator_lib.NoiseGenerator(
        noise_type="", noise_stddev=0.0, rng_seed=RNG_SEED)
    fec = fec_lib.FEC(
        nasbench=self._nasbench,
        noise_generator=noise_generator,
        mantissa_bits=24,
        hashing_time=10.0,
        max_num_evals=2,
        remove_probability=0.0,
        rng_seed=RNG_SEED)

    self._model_spec.graph_hash = "28cfc7874f6d200472e1a9dcd8650aa0"
    _ = fec.get_eval_outputs(model_spec=self._model_spec)
    _ = fec.get_eval_outputs(model_spec=self._model_spec)

    cache_stats = fec.get_cache_statistics()
    self.assertEqual(cache_stats["num_cache_misses"], 1)
    self.assertEqual(cache_stats["num_cache_partial_hits"], 1)
    self.assertEqual(cache_stats["num_cache_full_hits"], 0)
    self.assertEqual(cache_stats["cache_size"], 1)

  def test_get_cache_statistics_cache_full_hit(self):
    noise_generator = noise_generator_lib.NoiseGenerator(
        noise_type="", noise_stddev=0.0, rng_seed=RNG_SEED)
    fec = fec_lib.FEC(
        nasbench=self._nasbench,
        noise_generator=noise_generator,
        mantissa_bits=24,
        hashing_time=10.0,
        max_num_evals=2,
        remove_probability=0.0,
        rng_seed=RNG_SEED)

    self._model_spec.graph_hash = "28cfc7874f6d200472e1a9dcd8650aa0"
    _ = fec.get_eval_outputs(model_spec=self._model_spec)
    _ = fec.get_eval_outputs(model_spec=self._model_spec)
    _ = fec.get_eval_outputs(model_spec=self._model_spec)

    cache_stats = fec.get_cache_statistics()
    self.assertEqual(cache_stats["num_cache_misses"], 1)
    self.assertEqual(cache_stats["num_cache_partial_hits"], 1)
    self.assertEqual(cache_stats["num_cache_full_hits"], 1)
    self.assertEqual(cache_stats["cache_size"], 1)

  def test_get_cache_statistics_cache_hit_forgotten_hash(self):
    noise_generator = noise_generator_lib.NoiseGenerator(
        noise_type="", noise_stddev=0.0, rng_seed=RNG_SEED)
    fec = fec_lib.FEC(
        nasbench=self._nasbench,
        noise_generator=noise_generator,
        mantissa_bits=24,
        hashing_time=10.0,
        max_num_evals=1,
        remove_probability=1.0,
        rng_seed=RNG_SEED)

    self._model_spec.graph_hash = "28cfc7874f6d200472e1a9dcd8650aa0"
    # Cache miss.
    _ = fec.get_eval_outputs(model_spec=self._model_spec)
    # Cache hit and forgotten.
    _ = fec.get_eval_outputs(model_spec=self._model_spec)

    cache_stats = fec.get_cache_statistics()
    self.assertEqual(cache_stats["num_cache_misses"], 1)
    self.assertEqual(cache_stats["num_cache_partial_hits"], 0)
    self.assertEqual(cache_stats["num_cache_full_hits"], 1)
    self.assertEqual(cache_stats["cache_size"], 0)

  def test_get_cache_statistics_multiple_cache_hits_interleaved(self):
    noise_generator = noise_generator_lib.NoiseGenerator(
        noise_type="", noise_stddev=0.0, rng_seed=RNG_SEED)
    fec = fec_lib.FEC(
        nasbench=self._nasbench,
        noise_generator=noise_generator,
        mantissa_bits=24,
        hashing_time=10.0,
        max_num_evals=1,
        remove_probability=0.0,
        rng_seed=RNG_SEED)

    # Should be fresh.
    cache_stats = fec.get_cache_statistics()
    self.assertEqual(cache_stats["num_cache_misses"], 0)
    self.assertEqual(cache_stats["num_cache_partial_hits"], 0)
    self.assertEqual(cache_stats["num_cache_full_hits"], 0)
    self.assertEqual(cache_stats["cache_size"], 0)

    # Model A cache miss.
    self._model_spec.graph_hash = "28cfc7874f6d200472e1a9dcd8650aa0"
    _ = fec.get_eval_outputs(model_spec=self._model_spec)
    cache_stats = fec.get_cache_statistics()
    self.assertEqual(cache_stats["num_cache_misses"], 1)
    self.assertEqual(cache_stats["num_cache_partial_hits"], 0)
    self.assertEqual(cache_stats["num_cache_full_hits"], 0)
    self.assertEqual(cache_stats["cache_size"], 1)

    # Model B cache miss.
    self._model_spec.graph_hash = "1"
    _ = fec.get_eval_outputs(model_spec=self._model_spec)
    cache_stats = fec.get_cache_statistics()
    self.assertEqual(cache_stats["num_cache_misses"], 2)
    self.assertEqual(cache_stats["num_cache_partial_hits"], 0)
    self.assertEqual(cache_stats["num_cache_full_hits"], 0)
    self.assertEqual(cache_stats["cache_size"], 2)

    # Model C cache miss.
    self._model_spec.graph_hash = "2"
    _ = fec.get_eval_outputs(model_spec=self._model_spec)
    cache_stats = fec.get_cache_statistics()
    self.assertEqual(cache_stats["num_cache_misses"], 3)
    self.assertEqual(cache_stats["num_cache_partial_hits"], 0)
    self.assertEqual(cache_stats["num_cache_full_hits"], 0)
    self.assertEqual(cache_stats["cache_size"], 3)

    # Model A again, now cache hit.
    self._model_spec.graph_hash = "28cfc7874f6d200472e1a9dcd8650aa0"
    _ = fec.get_eval_outputs(model_spec=self._model_spec)
    cache_stats = fec.get_cache_statistics()
    self.assertEqual(cache_stats["num_cache_misses"], 3)
    self.assertEqual(cache_stats["num_cache_partial_hits"], 0)
    self.assertEqual(cache_stats["num_cache_full_hits"], 1)
    self.assertEqual(cache_stats["cache_size"], 3)

    # Model C again, now cache hit.
    self._model_spec.graph_hash = "2"
    _ = fec.get_eval_outputs(model_spec=self._model_spec)
    cache_stats = fec.get_cache_statistics()
    self.assertEqual(cache_stats["num_cache_misses"], 3)
    self.assertEqual(cache_stats["num_cache_partial_hits"], 0)
    self.assertEqual(cache_stats["num_cache_full_hits"], 2)
    self.assertEqual(cache_stats["cache_size"], 3)

  def test_get_cache_statistics_hashes_correctly(self):
    noise_generator = noise_generator_lib.NoiseGenerator(
        noise_type="", noise_stddev=0.0, rng_seed=RNG_SEED)
    mantissa_bits = 24
    hashing_time = 10.0
    fec = fec_lib.FEC(
        nasbench=self._nasbench,
        noise_generator=noise_generator,
        mantissa_bits=mantissa_bits,
        hashing_time=hashing_time,
        max_num_evals=1,
        remove_probability=0.0,
        save_history=True,
        rng_seed=RNG_SEED)

    _ = fec.get_eval_outputs(model_spec=self._model_spec)

    cache_stats = fec.get_cache_statistics()

    hasher = hasher_lib.Hasher(
        nasbench=self._nasbench,
        mantissa_bits=mantissa_bits,
        hashing_time=hashing_time)
    expected_model_hash, _ = hasher.get_unified_functional_hash(
        self._model_spec)

    self.assertEqual(cache_stats["num_cache_misses"], 1)
    self.assertEqual(cache_stats["num_cache_partial_hits"], 0)
    self.assertEqual(cache_stats["num_cache_full_hits"], 0)
    self.assertEqual(cache_stats["cache_size"], 1)
    self.assertEqual(cache_stats["model_hashes"], str(expected_model_hash))


if __name__ == "__main__":
  absltest.main()
