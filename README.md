# Unified Functional Hashing

## Overview
This directory contains the publicly available [Colab Notebook](https://colab.research.google.com/github/google-research/unified_functional_hashing/blob/main/Unified_Functional_Hashing.ipynb) for the paper:

[**Unified Functional Hashing**](
TODO:{link to arxiv}).

Ryan Gillard, Stephen Jonany, Yingjie Miao, Michael Munn, Connal de Souza, Jonathan Dungay, Chen Liang, David R. So, Quoc V. Le, Esteban Real (2022).


## Available Features

With this notebook you will be able to:

* Run experiments on the NASBench101 dataset with various setups
  * With/without the Functional Equivalence Cache (FEC) described in the paper
  * Using Regularized Evolution, Elitism, or Random Search algorithms
  * Adding noise to evaluation as desired
* Examine the results of each experiment
  * Viewing and plotting the test and validation fitnesses versus the experiment progress (time or evaluations completed)
  * Inspecting various statistics about evaluations and the FEC performance

## Citation

If you find this code or paper useful, please cite:

```
@article{uid,
title = {Unified Functional Hashing in Automatic Machine Learning},
author = {Ryan Gillard and Stephen Jonany and Yingjie Miao and Michael Munn and Connal de Souza and Jonathan Dungay and Chen Liang and David R. So and Quoc V. Le and Esteban Real},
journal = {-}
}
```

[![Unittests](https://github.com/google-research/functional_equivalence/actions/workflows/pytest_and_autopublish.yml/badge.svg)](https://github.com/google-research/functional_equivalence/actions/workflows/pytest_and_autopublish.yml)
[![PyPI version](https://badge.fury.io/py/functional_equivalence.svg)](https://badge.fury.io/py/functional_equivalence)

*This is not an officially supported Google product.*
