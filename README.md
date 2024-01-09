# Activation Based Pruning of Neural Networks

We present a novel technique for pruning called **activation-based** pruning to effectively prune fully connected feedforward neural networks for multi-object classification. 
Our technique is based on the number of times each neuron is activated during model training. We compare the performance of activation-based pruning with a popular pruning method: 
magnitude-based pruning. Further analysis demonstrates that activation-based pruning can be considered a dimensionality reduction technique as it leads to a low-rank matrix 
approximation for each hidden layer of the neural network. We also demonstrate that the rank-reduced neural network generated using activation-based pruning has better test 
results than a rank-reduced network using principal component analysis. We provide empirical results to show that after each successive pruning the amount of reduction in the 
magnitude of singular values of each matrix representing the hidden layers of network is equivalent to introducing the sum of singular values of the hidden layers as a 
regularization parameter to the objective function.

# Dependencies:
The code has been tested for the following software versions: 
* anaconda - $2022.10$ 
* python -  $3.9.15$
* tensorflow - $2.10.0$
* pandas - $1.5.1$
* matplotlib - $3.5.3$ 
* seaborn - $0.12.1$ 
* openpyxl - $3.0.10$.

Datasets:
* [MNIST](https://www.tensorflow.org/datasets/catalog/mnist)
* [Fashion MNIST](https://www.tensorflow.org/datasets/catalog/fashion_mnist)
