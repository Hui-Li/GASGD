# GASGD
C++ implementation of GASGD presented in the following paper:

*F. Petroni and L. Querzoni. GASGD: stochastic gradient descent for distributed asynchronous matrix completion via graph partitioning. In RecSys, pages 241–248, 2014.*
   
Java simulator provided by original authors: https://github.com/fabiopetroni/GASGD

### Environment

- Ubuntu 16.04
- CMake 2.8
- GCC 5.4
- MPICH 3.1.4
- Boost 1.63
- Intel TBB 4.4~20151115

### Parameters

Default values are used in Java simulator from original authors.

- `k`: dimensionality of latent vector (Default 10).
- `lr`: learning rate (Default 0.01).
- `lambda`: regularization weight (Default 0.05).
- `folder`: how many times the machines communicate during each epoch (Default 1)
- `max_iter`: how many iterations to be performed by the sgd algorithm (Default 30)
- `node`: number of machines (Default 1)
- `thread`: number of thread per machine (Default 4)
- `g_period`: synchronization window for graph partitioning (Default 0.01)
- `path`: file path to the folder of data which contains meta and CSR file
- `verbose`: whether the program should output information for debugging (Default true).

### Data
Our implementation uses [CSR](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_.28CSR.2C_CRS_or_Yale_format.29) format as input. Additionally, there should be a meta file with the following format:
```
69878 10677
7972661 train.dat
2027393 test.dat
```
where 69878 is the number of users, 10677 is the number of items, 7972661 is the number of training ratings, train.dat is the path to training file (in CSR format), 2027393 is the number of testing ratings and test.dat is the path to testing file (in CSR format).

You can use our tool [MFDataTransform](https://github.com/Hui-Li/MFDataTransform) to transform public datasets to CSR format.

### Example
`runGASGD.sh` provides an example for running the algorithm.
