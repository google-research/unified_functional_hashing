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

"""Constants for NASBench.
"""

INPUT = "input"
OUTPUT = "output"
CONV3X3 = "conv3x3-bn-relu"
CONV1X1 = "conv1x1-bn-relu"
MAXPOOL3X3 = "maxpool3x3"
NUM_VERTICES = 7
MAX_EDGES = 9
EDGE_SPOTS = NUM_VERTICES * (NUM_VERTICES - 1) / 2   # Upper triangular matrix
OP_SPOTS = NUM_VERTICES - 2   # Input/output vertices are fixed
ALLOWED_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]
ALLOWED_EDGES = [0, 1]   # Binary adjacency matrix

NASBENCH_CONFIG = {
    "available_ops": [
        "conv3x3-bn-relu", "conv1x1-bn-relu", "maxpool3x3"
    ],
    "nasbench_batch_size": 256,
    "nasbench_data_format": "channels_last",
    "nasbench_intermediate_evaluations": ["0.5"],
    "nasbench_learning_rate": 0.1,
    "nasbench_lr_decay_method": "COSINE_BY_STEP",
    "nasbench_max_attempts": 5,
    "nasbench_max_edges": 9,
    "nasbench_module_vertices": 7,
    "nasbench_momentum": 0.9,
    "nasbench_num_labels": 10,
    "nasbench_num_modules_per_stack": 3,
    "nasbench_num_repeats": 3,
    "nasbench_num_stacks": 3,
    "nasbench_sample_data_file": "",
    "nasbench_stem_filter_size": 128,
    "nasbench_test_data_file": "",
    "nasbench_tpu_iterations_per_loop": 100,
    "nasbench_tpu_num_shards": 2,
    "nasbench_train_data_files": [],
    "nasbench_train_epochs": 108,
    "nasbench_train_seconds": 14400.0,
    "nasbench_use_tpu": True,
    "nasbench_valid_data_file": "",
    "nasbench_weight_decay": 0.0001
}

MAX_RNG_SEED = 2147483647
