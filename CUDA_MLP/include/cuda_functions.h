#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H

#include <vector>
using namespace std;
// #include <cuda_runtime.h>

__device__ double sigmoid(double x);
__device__ double d_sigmoid(double x);
__device__ double softmax(double x);
__device__ double d_softmax_cross_entropy(double y, double y_hat);

double random(double min, double max);
std::vector<double> matrix_to_array(const std::vector<std::vector<double>> &matrix)

#endif