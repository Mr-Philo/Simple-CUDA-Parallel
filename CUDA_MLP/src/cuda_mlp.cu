#include "cuda_mlp.h"
#include <cuda_runtime.h>
using namespace std;

MLP_CUDA::MLP_CUDA(int input_dim, int hidden_dim, int output_dim) {
    input_dim = input_dim;
    hidden_dim = hidden_dim;
    output_dim = output_dim;

    // Randomly initialize the weights and biases
    d_W1 = new double[hidden_dim * input_dim];
    d_W2 = new double[output_dim * hidden_dim];
    d_b1 = new double[hidden_dim];
    d_b2 = new double[output_dim];

    // initialize W1, W2, b1, b2
    // no need to move to GPU, since the time consuming of transfering data is much more than the time of initializing
    for(int i = 0;i<hidden_dim;i++){
        for (int j = 0; j < input_dim; ++j) {
            d_W1[i * input_dim + j] = random(-1,1);
        }
    }
    for(int i = 0;i<output_dim;i++){
        for (int j = 0; j < hidden_dim; ++j) {
            d_W2[i * hidden_dim + j] = random(-1,1);
        }
    }
    for(int i = 0; i<hidden_dim; i++) {
        d_b1[i] = random(-1,1);
    }
    for (int i = 0; i < output_dim; ++i) {
        d_b2[i] = random(-1,1);
    }

    // Initialize the gradients
    d_W1_grad = new double[hidden_dim * input_dim];
    d_W2_grad = new double[output_dim * hidden_dim];
    d_b1_grad = new double[hidden_dim];
    d_b2_grad = new double[output_dim];
}

__device__ void MLP_CUDA::zero_grad() {
    // fill the gradients with 0
    for (int i = 0; i < hidden_dim * input_dim; ++i) {
        d_W1_grad[i] = 0;
    }
    for (int i = 0; i < output_dim * hidden_dim; ++i) {
        d_W2_grad[i] = 0;
    }
    for (int i = 0; i < hidden_dim; ++i) {
        d_b1_grad[i] = 0;
    }
    for (int i = 0; i < output_dim; ++i) {
        d_b2_grad[i] = 0;
    }
}


// TODO: 更新：将CUDA内核缩减至一个，只在main.cpp函数中执行，其中整个流程的主要函数采取_device__的方式，即在CUDA内核中执行
// forward CUDA kernel
__global__ void forward_kernel(double* input, double* W1, double* y1, double* b1, double* z1, double* W2, double* y2, double* b2, double* z2, int input_dim, int hidden_dim, int output_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // input -> first hidden layer
    if (idx < hidden_dim) {
        double sum = 0;
        for (int j = 0; j < input_dim; ++j) {
            sum += W1[idx * input_dim + j] * input[j];  // matrix-vector multiplication
        }
        y1[idx] = sum + b1[idx];        // vector add
        z1[idx] = sigmoid(y1[idx]);     // activation function
    }
    // first hidden layer -> output
    if (idx < output_dim) {
        double sum = 0;
        for (int j = 0; j < hidden_dim; ++j) {
            sum += W2[idx * hidden_dim + j] * z1[j];    // matrix-vector multiplication
        }
        y2[idx] = sum + b2[idx];        // vector add
        z2[idx] = softmax(y2[idx]);     // activation function
        softmax_kernel(z2, output_dim);  // softmax
    }
}

vector<double> MLP_CUDA::forward(const vector<unsigned char> &x) {
    input = vector<double>(x.begin(),x.end());
    y1 = vector<double>(hidden_dim, 0);
    z1 = vector<double>(hidden_dim, 0);
    y2 = vector<double>(output_dim, 0);
    z2 = vector<double>(output_dim, 0);

    double* d_input;
    double* d_W1;
    double* d_y1;
    double* d_b1;
    double* d_z1;
    double* d_W2;
    double* d_y2;
    double* d_b2;
    double* d_z2;

    cudaMalloc((void**)&d_input, input.size() * sizeof(double));
    cudaMalloc((void**)&d_W1, W1.size() * sizeof(double));
    cudaMalloc((void**)&d_y1, y1.size() * sizeof(double));
    cudaMalloc((void**)&d_b1, b1.size() * sizeof(double));
    cudaMalloc((void**)&d_z1, z1.size() * sizeof(double));
    cudaMalloc((void**)&d_W2, W2.size() * sizeof(double));
    cudaMalloc((void**)&d_y2, y2.size() * sizeof(double));
    cudaMalloc((void**)&d_b2, b2.size() * sizeof(double));
    cudaMalloc((void**)&d_z2, z2.size() * sizeof(double));

    cudaMemcpy(d_input, input.data(), input.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, W1.data(), W1.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y1, y1.data(), y1.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, b1.data(), b1.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z1, z1.data(), z1.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, W2.data(), W2.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y2, y2.data(), y2.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, b2.data(), b2.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z2, z2.data(), z2.size() * sizeof(double), cudaMemcpyHostToDevice);

    int block_size = 256;
    int num_blocks = (hidden_dim + block_size - 1) / block_size;
    forward_kernel<<<num_blocks, block_size>>>(d_input, d_W1, d_y1, d_b1, d_z1, d_W2, d_y2, d_b2, d_z2, input_dim, hidden_dim, output_dim);

    cudaMemcpy(y1.data(), d_y1, y1.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(z1.data(), d_z1, z1.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(y2.data(), d_y2, y2.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(z2.data(), d_z2, z2.size() * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_W1);
    cudaFree(d_y1);
    cudaFree(d_b1);
    cudaFree(d_z1);
    cudaFree(d_W2);
    cudaFree(d_y2);
    cudaFree(d_b2);
    cudaFree(d_z2);

    return z2;
}

__global__ void backward_kernel(double* b2_grad, double* W2_grad, double* b1_grad, double* W1_grad, const double* y, const double* y_hat, const double* z1, const double* input, int hidden_dim, int output_dim, int input_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_dim) {
        b2_grad[idx] = d_softmax_cross_entropy(y[idx], y_hat[idx]);
        for (int j = 0; j < hidden_dim; ++j) {
            W2_grad[idx * hidden_dim + j] = b2_grad[idx] * z1[j];
        }
    }
    if (idx < hidden_dim) {
        double sum = 0;
        for (int i = 0; i < output_dim; ++i) {
            sum += W2[i][idx] * b2_grad[i];
        }
        b1_grad[idx] = sum * d_sigmoid(y1[idx]);
        for (int j = 0; j < input_dim; ++j) {
            W1_grad[idx * input_dim + j] = b1_grad[idx] * input[j];
        }
    }
}

void MLP_CUDA::backward(const vector<double> &y, const vector<double> &y_hat) {
    double* d_b2_grad;
    double* d_W2_grad;
    double* d_b1_grad;
    double* d_W1_grad;
    double* d_y;
    double* d_y_hat;
    double* d_z1;
    double* d_input;

    cudaMalloc((void**)&d_b2_grad, b2_grad.size() * sizeof(double));
    cudaMalloc((void**)&d_W2_grad, W2_grad.size() * sizeof(double));
    cudaMalloc((void**)&d_b1_grad, b1_grad.size() * sizeof(double));
    cudaMalloc((void**)&d_W1_grad, W1_grad.size() * sizeof(double));
    cudaMalloc((void**)&d_y, y.size() * sizeof(double));
    cudaMalloc((void**)&d_y_hat, y_hat.size() * sizeof(double));
    cudaMalloc((void**)&d_z1, z1.size() * sizeof(double));
    cudaMalloc((void**)&d_input, input.size() * sizeof(double));

    cudaMemcpy(d_b2_grad, b2_grad.data(), b2_grad.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2_grad, W2_grad.data(), W2_grad.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1_grad, b1_grad.data(), b1_grad.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1_grad, W1_grad.data(), W1_grad.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y.data(), y.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_hat, y_hat.data(), y_hat.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z1, z1.data(), z1.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, input.data(), input.size() * sizeof(double), cudaMemcpyHostToDevice);

    int block_size = 256;
    int num_blocks = (output_dim + block_size - 1) / block_size;
    backward_kernel<<<num_blocks, block_size>>>(d_b2_grad, d_W2_grad, d_b1_grad, d_W1_grad, d_y, d_y_hat, d_z1, d_input, hidden_dim, output_dim, input_dim);

    cudaMemcpy(b2_grad.data(), d_b2_grad, b2_grad.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(W2_grad.data(), d_W2_grad, W2_grad.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(b1_grad.data(), d_b1_grad, b1_grad.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(W1_grad.data(), d_W1_grad, W1_grad.size() * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_b2_grad);
    cudaFree(d_W2_grad);
    cudaFree(d_b1_grad);
    cudaFree(d_W1_grad);
    cudaFree(d_y);
    cudaFree(d_y_hat);
    cudaFree(d_z1);
    cudaFree(d_input);
}

__global__ void update_kernel(double* weights, double* gradients, double lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= lr * gradients[idx];
    }
}

void MLP_CUDA::update(double lr) {
    double* d_W1;
    double* d_W1_grad;
    double* d_b1;
    double* d_b1_grad;
    double* d_W2;
    double* d_W2_grad;
    double* d_b2;
    double* d_b2_grad;

    cudaMalloc((void**)&d_W1, W1.size() * sizeof(double));
    cudaMalloc((void**)&d_W1_grad, W1_grad.size() * sizeof(double));
    cudaMalloc((void**)&d_b1, b1.size() * sizeof(double));
    cudaMalloc((void**)&d_b1_grad, b1_grad.size() * sizeof(double));
    cudaMalloc((void**)&d_W2, W2.size() * sizeof(double));
    cudaMalloc((void**)&d_W2_grad, W2_grad.size() * sizeof(double));
    cudaMalloc((void**)&d_b2, b2.size() * sizeof(double));
    cudaMalloc((void**)&d_b2_grad, b2_grad.size() * sizeof(double));

    cudaMemcpy(d_W1, W1.data(), W1.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1_grad, W1_grad.data(), W1_grad.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, b1.data(), b1.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1_grad, b1_grad.data(), b1_grad.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, W2.data(), W2.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2_grad, W2_grad.data(), W2_grad.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, b2.data(), b2.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2_grad, b2_grad.data(), b2_grad.size() * sizeof(double), cudaMemcpyHostToDevice);

    int block_size = 256;
    int num_blocks = (W1.size() + block_size - 1) / block_size;
    update_kernel<<<num_blocks, block_size>>>(d_W1, d_W1_grad, lr, W1.size());

    num_blocks = (b1.size() + block_size - 1) / block_size;
    update_kernel<<<num_blocks, block_size>>>(d_b1, d_b1_grad, lr, b1.size());

    num_blocks = (W2.size() + block_size - 1) / block_size;
    update_kernel<<<num_blocks, block_size>>>(d_W2, d_W2_grad, lr, W2.size());

    num_blocks = (b2.size() + block_size - 1) / block_size;
    update_kernel<<<num_blocks, block_size>>>(d_b2, d_b2_grad, lr, b2.size());

    cudaMemcpy(W1.data(), d_W1, W1.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(b1.data(), d_b1, b1.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(W2.data(), d_W2, W2.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(b2.data(), d_b2, b2.size() * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_W1);
    cudaFree(d_W1_grad);
    cudaFree(d_b1);
    cudaFree(d_b1_grad);
    cudaFree(d_W2);
    cudaFree(d_W2_grad);
    cudaFree(d_b2);
    cudaFree(d_b2_grad);
}

MLP_CUDA::~MLP_CUDA() = default;
