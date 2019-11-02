# Project
## Current Progress: 
have create custome dataset of MNIST and CIFAR10 in different distrubution and test the performance of different distribution, haven't get a baseline yet

## TODO:
1. create more skewed distribution (current distribution doesn't affect the performance a lot)
2. analysis the effect of skewed distributions on percision and recall for each class
3. create a baseline on each distribution

## Description of important files

### CustomDataset.py
Contain classes and methods to load custume datasets in given distributions (index files).

### Indexes_generator.ipynb
Randomly select indexes of the original dataset based on the given distribution and save the selected indexes as ".npy" file.

### Baseline.py (in progress)
Testing performance on different distribution.
