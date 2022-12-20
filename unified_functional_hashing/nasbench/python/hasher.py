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

"""UnifiedFunctionalHasher for NASBench."""
import math
from typing import Tuple

from nasbench import api


class Hasher():
  """UnifiedFunctionalHasher for NASBench."""

  def __init__(self,
               nasbench: api.NASBench,
               mantissa_bits: int = 24,
               hashing_time: float = 10.0):
    """Initializes UnifiedFunctionalHasher.

    Args:
      nasbench: NASBench instance.
      mantissa_bits: Number of bits to use in the mantissa.
      hashing_time: Number of seconds it takes to generate hash.
    """
    self.nasbench = nasbench
    self.mantissa_bits = mantissa_bits
    self.hashing_time = hashing_time
    self.accuracy_list = [
        "final_train_accuracy", "halfway_train_accuracy",
        "final_validation_accuracy", "halfway_validation_accuracy"
    ]

  def significant_float_mix(self, accuracies: list[float],
                            mantissa_bits: int) -> int:
    """Mixes bits in a list of floats, rounded according to mantissa_bits.

    Args:
      accuracies: list of floats to mix, which represent accuracies in this
        context.
      mantissa_bits: Number of bits to use in the mantissa.

    Returns:
      An integer produced by mixing the given list of floats.
    """

    def rotate_left(num: int, bits: int) -> int:
      bit = num & (1 << (bits - 1))
      num <<= 1
      if bit:
        num |= 1
      num &= (2**bits - 1)

      return int(num)

    def interpret_float_as_int(val: float, bits: int) -> int:
      mantissa, exponent = math.frexp(val)
      return int(math.ldexp(round(mantissa, bits), exponent) * 1 * 10**(bits))

    def mix_bits(mix: int, val: float, bits: int) -> int:
      int_val = interpret_float_as_int(val, bits)

      v1 = mix ^ rotate_left(int_val, 40)
      v2 = int_val ^ rotate_left(mix, 39)
      v3 = v1 * v2
      return int((v3 ^ (v3 >> 11)) % (2**63))  # limit to 64 bit signed int

    mix = 0
    for accuracy in accuracies:
      mantissa, exponent = math.frexp(accuracy)
      mantissa = round(mantissa, mantissa_bits)
      sign = 0
      mix = mix_bits(mix, sign, mantissa_bits)
      mix = mix_bits(mix, exponent, mantissa_bits)
      mix = mix_bits(mix, mantissa, mantissa_bits)

    return mix

  def get_unified_functional_hash(self,
                                  model_spec: api.ModelSpec,
                                  test: bool = False) -> Tuple[int, float]:
    """Calculates unified functional hash from model spec.

    Args:
      model_spec: ModelSpec matrix.
      test: Whether called from a test or not.

    Returns:
      Integer hash and float time to hash.
    """
    if self.nasbench.is_valid(model_spec):
      if test and not hasattr(model_spec, "graph_hash"):
        model_spec.graph_hash = model_spec.hash_spec(
            canonical_ops=self.nasbench.config["nasbench_available_ops"])
      _, computed_stats = self.nasbench.get_metrics_from_spec(model_spec)
    epoch_stats = computed_stats[4][0]
    accuracies = [epoch_stats[accuracy] for accuracy in self.accuracy_list]
    return (self.significant_float_mix(accuracies,
                                       self.mantissa_bits),
            self.hashing_time)
