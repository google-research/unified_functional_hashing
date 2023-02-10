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
import hashlib
import struct
from typing import List, Tuple

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
      hashing_time: Simulated number of seconds it takes to generate hash.
    """
    self.nasbench = nasbench
    if not (mantissa_bits <= 52 and mantissa_bits > 0):
      raise ValueError(
          f"mantissa_bits is set to {mantissa_bits}, but must be between 1 and"
          " 52 for double-precision floats."
      )
    self.mantissa_bits = mantissa_bits
    self.exponent_bitmask = int("0" + 11 * "1" + 52 * "0", 2)
    self.truncated_mantissa_bitmask = int(
        12 * "0" + mantissa_bits * "1" + (52 - mantissa_bits) * "0", 2
    )
    self.hashing_time = hashing_time
    self.accuracy_list = [
        "final_train_accuracy", "halfway_train_accuracy",
        "final_validation_accuracy", "halfway_validation_accuracy"
    ]

  def significant_float_mix(self, accuracies: List[float]) -> int:
    """Mixes bits in a list of floats, rounded according to mantissa_bits.

    Args:
      accuracies: list of floats to mix, which represent accuracies in this
        context.

    Returns:
      An integer produced by mixing the given list of floats.
    """
    def hash_to_integer(value: Tuple[int, int]) -> int:
      """Return a 64-bit integer representing the hash of the given value."""
      hasher = hashlib.sha256()
      hasher.update(bytes(str(value), "utf8"))
      return int.from_bytes(hasher.digest(), "little", signed=False) % (2**63)

    def get_float_bits(val: float, bit_mask: int) -> int:
      """Apply the bit_mask to the float val, and reinterpret it as an int."""
      return struct.unpack("<Q", struct.pack("<d", val))[0] & bit_mask

    def mix_bits(mix: int, val: float, bit_mask: int) -> int:
      """Returns an integer formed by mixing mix (existing) and val(new)."""
      return hash_to_integer((mix, get_float_bits(val, bit_mask)))

    mix = 0
    for accuracy in accuracies:
      # All accuracies are positive,
      # so we do not bother mixing in the sign bit which is always 0.
      mix = mix_bits(mix, accuracy, self.exponent_bitmask)
      mix = mix_bits(mix, accuracy, self.truncated_mantissa_bitmask)

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
            canonical_ops=self.nasbench.config["available_ops"])
      _, computed_stats = self.nasbench.get_metrics_from_spec(model_spec)
    epoch_stats = computed_stats[4][0]
    accuracies = [epoch_stats[accuracy] for accuracy in self.accuracy_list]
    return (self.significant_float_mix(accuracies), self.hashing_time)
