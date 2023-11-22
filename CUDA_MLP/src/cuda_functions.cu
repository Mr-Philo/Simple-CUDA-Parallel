#include "cuda_functions.h"
#include <cmath>
#include <cassert>
#include <cuda_runtime.h>

// double cross_entropy(const double* y, const double* y_hat, int size) {
//     double loss = 0;
//     for (int i = 0; i < size; i++) {
//         loss -= y[i] * log(y_hat[i]);
//     }
//     return loss;
// }
// // the following function(which uses atomicAdd) is probaly slower than sequential version, or even not working due to 'atomicAdd' does not support double type on GPU capability < 6.0, so we may not use it.
// __global__ void cross_entropy_kernel(const double* y, const double* y_hat, double* loss, int size) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid < size) {
//         atomicAdd(loss, -y[tid] * log(y_hat[tid]));
//     }
// }

__device__ double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

__device__ double d_sigmoid(double x) {
    double sigmoid_x = sigmoid(x);
    return sigmoid_x * (1 - sigmoid_x);
}

__device__ double softmax(double x) {
    return exp(x);
}

__device__ double d_softmax_cross_entropy(double y, double y_hat) {
    return y_hat - y;
}

// non-CUDA functions
double random(double min, double max) {
    return min + (max - min) * rand() / (RAND_MAX + 1.0);
}

// change matrix to array (2D matrix can not be passed to GPU)
std::vector<float> matrix_to_array(const std::vector<std::vector<float>> &matrix) {  
  std::vector<float> array;  
  for (const auto &row : matrix) {  
    array.insert(array.end(), row.begin(), row.end());  
  }  
  return array;  
}  
