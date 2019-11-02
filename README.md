# Project
## Current Progress: 
Have created custome datasets of MNIST and CIFAR10 in different distrubutions and tested the performance of different distributions, haven't get a baseline yet.

## TODO:
1. Create more skewed distribution (current skewed distributions doesn't affect the performance a lot)
2. Analysis the effect of skewed distributions on percision and recall of each class
3. Create a baseline on each distribution

## Descriptions of Important Files

### CustomDataset.py
Contain classes and methods to load custom datasets in given distributions (index files).

### Indexes_generator.ipynb
Randomly select indexes of the original dataset based on the given distribution and save the selected indexes as ".npy" file.

### Baseline.py (in progress)
Testing performance on different distribution.
