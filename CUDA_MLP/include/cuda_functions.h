#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H

#include <vector>
using namespace std;
// #include <cuda_runtime.h>
__device__ double sigmoid(double x);
__global__ void sigmoid_kernel(double* x, int size);
__device__ double d_sigmoid(double x);
__global__ void d_sigmoid_kernel(double* x, double* d, int size);
__device__ double softmax(double x);
__global__ void softmax_kernel(double* x, double* ex, double sum, int size);
__global__ void normalize_kernel(double* ex, double sum, int size);
__device__ double d_softmax_cross_entropy(double y, double y_hat);
__global__ void d_softmax_cross_entropy_kernel(double* y, double* y_hat, double* d, int size);
__global__ void outer_product_kernel(double* x, double* y, double* result, int x_size, int y_size);
__global__ void matrix_add_kernel(double* x, double* y, int rows, int cols);
__global__ void vector_add_kernel(double* x, double* y, int size);
__global__ void matrix_multiply_kernel(double* x, double y, int rows, int cols);
__global__ void vector_multiply_kernel(double* x, double y, int size);
__global__ void matrix_dot_kernel(double* a, double* b, double* result, int a_rows, int a_cols, int b_cols);
__global__ void vector_dot_kernel(double* x, double* y, double* result, int size);
__global__ void transpose_kernel(double* x, double* result, int m, int n);

double random(double min, double max);

#endif