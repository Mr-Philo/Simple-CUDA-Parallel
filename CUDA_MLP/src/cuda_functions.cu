#include "cuda_functions.h"
#include <cmath>
#include <cassert>
#include <cuda_runtime.h>

using namespace std;

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

/*double cross_entropy(const vector<double>& y, const vector<double>& y_hat) {
    int size = y.size();
    double* d_y;
    double* d_y_hat;
    double* d_loss;
    double loss = 0;

    // Allocate memory on the device
    cudaMalloc((void**)&d_y, size * sizeof(double));
    cudaMalloc((void**)&d_y_hat, size * sizeof(double));
    cudaMalloc((void**)&d_loss, sizeof(double));

    // Copy input data from host to device
    cudaMemcpy(d_y, y.data(), size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_hat, y_hat.data(), size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_loss, &loss, sizeof(double), cudaMemcpyHostToDevice);

    // Launch the kernel
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    cross_entropy_kernel<<<gridSize, blockSize>>>(d_y, d_y_hat, d_loss, size);

    // Copy the result back to host
    cudaMemcpy(&loss, d_loss, sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_y);
    cudaFree(d_y_hat);
    cudaFree(d_loss);

    return loss;
}*/

__device__ double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

__global__ void sigmoid_kernel(double* x, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        x[tid] = sigmoid(x[tid]);
    }
}

__device__ double d_sigmoid(double x) {
    double sigmoid_x = sigmoid(x);
    return sigmoid_x * (1 - sigmoid_x);
}

__global__ void d_sigmoid_kernel(double* x, double* d, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        d[tid] = d_sigmoid(x[tid]);
    }
}

__device__ double softmax(double x) {
    return exp(x);
}

__global__ void softmax_kernel(double* x, double* ex, double sum, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        ex[tid] = softmax(x[tid]);
        atomicAdd(&sum, ex[tid]);
    }
}

__global__ void normalize_kernel(double* ex, double sum, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        ex[tid] /= sum;
    }
}

__device__ double d_softmax_cross_entropy(double y, double y_hat) {
    return y_hat - y;
}

__global__ void d_softmax_cross_entropy_kernel(double* y, double* y_hat, double* d, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        d[tid] = d_softmax_cross_entropy(y[tid], y_hat[tid]);
    }
}

__global__ void outer_product_kernel(double* x, double* y, double* result, int x_size, int y_size) {
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (tid_x < x_size && tid_y < y_size) {
        result[tid_x * y_size + tid_y] = x[tid_x] * y[tid_y];
    }
}

__global__ void matrix_add_kernel(double* x, double* y, int rows, int cols) {
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (tid_x < rows && tid_y < cols) {
        x[tid_x * cols + tid_y] += y[tid_x * cols + tid_y];
    }
}

__global__ void vector_add_kernel(double* x, double* y, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        x[tid] += y[tid];
    }
}

__global__ void matrix_multiply_kernel(double* x, double y, int rows, int cols) {
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (tid_x < rows && tid_y < cols) {
        x[tid_x * cols + tid_y] *= y;
    }
}

__global__ void vector_multiply_kernel(double* x, double y, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        x[tid] *= y;
    }
}

__global__ void matrix_dot_kernel(double* a, double* b, double* result, int a_rows, int a_cols, int b_cols) {
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid_x < a_rows) {
        double sum = 0;
        for (int j = 0; j < a_cols; j++) {
            sum += a[tid_x * a_cols + j] * b[j];
        }
        result[tid_x] = sum;
    }
}

__global__ void vector_dot_kernel(double* x, double* y, double* result, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        result[tid] = x[tid] * y[tid];
    }
}

__global__ void transpose_kernel(double* x, double* result, int m, int n) {
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (tid_x < m && tid_y < n) {
        result[tid_y * m + tid_x] = x[tid_x * n + tid_y];
    }
}

// 调用实例
// vector<double> vector_dot(vector<double> x, vector<double> y) {
//     assert(x.size() == y.size());
//     int size = x.size();
//     double* d_x, *d_y, *d_result;
//     cudaMalloc((void**)&d_x, size * sizeof(double));
//     cudaMalloc((void**)&d_y, size * sizeof(double));
//     cudaMalloc((void**)&d_result, size * sizeof(double));

//     cudaMemcpy(d_x, x.data(), size * sizeof(double), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_y, y.data(), size * sizeof(double), cudaMemcpyHostToDevice);

//     int threadsPerBlock = 256;
//     int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
//     vector_dot_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_result, size);

//     vector<double> result(size);
//     cudaMemcpy(result.data(), d_result, size * sizeof(double), cudaMemcpyDeviceToHost);

//     cudaFree(d_x);
//     cudaFree(d_y);
//     cudaFree(d_result);

//     return result;
// }

// vector<vector<double>> transpose(vector<vector<double>> x) {
//     int m = x.size();
//     int n = x[0].size();
//     double* d_x, *d_result;
//     cudaMalloc((void**)&d_x, m * n * sizeof(double));
//     cudaMalloc((void**)&d_result, m * n * sizeof(double));

//     cudaMemcpy(d_x, x.data(), m * n * sizeof(double), cudaMemcpyHostToDevice);

//     dim3 threadsPerBlock(16, 16);
//     dim3 blocksPerGrid((m + threadsPerBlock.x - 1) / threadsPerBlock.x, (n + threadsPerBlock.y - 1) / threadsPerBlock.y);
//     transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_result, m, n);

//     vector<vector<double>> result(n, vector<double>(m));
//     cudaMemcpy(result.data(), d_result, m * n * sizeof(double), cudaMemcpyDeviceToHost);

//     cudaFree(d_x);
//     cudaFree(d_result);

//     return result;
// }

// non-CUDA functions
double random(double min, double max) {
    return min + (max - min) * rand() / (RAND_MAX + 1.0);
}
