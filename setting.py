train_max_size = 4000
evaluation_max_size = 1000

p_MNIST = [10, 25, 50, 100, 250, 500, 1000, 2000, 4000]
mu_MNIST = [0.2, 0.5, 0.8]

p_CIFAR10 = [2, 10, 20, 40]
mu_CIFAR10 = [0.2, 0.5, 0.8]

permutations = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [9, 2, 3, 4, 7, 8, 0, 6, 5, 1],
    [1, 8, 6, 9, 0, 3 , 7, 5, 4, 2],
    [5, 9, 2, 7, 8, 6, 0, 4, 1, 3],
    [4, 2, 0, 3, 1, 9, 7, 8, 6, 5]
]

LINEAR = "linear"
EXPONENTIAL = "exponential"
STEP = "step"