#include "cuda_mlp.h"
#include <cuda_runtime.h>
using namespace std;

// move the following functions from cuda_funtions.cu here
__device__ double cu_sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

__device__ double cu_d_sigmoid(double x) {
    double sigmoid_x = cu_sigmoid(x);
    return sigmoid_x * (1 - sigmoid_x);
}

__device__ double cu_softmax(double x) {
    return exp(x);
}

__device__ double cu_d_softmax_cross_entropy(double y, double y_hat) {
    return y_hat - y;
}

// non-CUDA functions
double cu_random(double min, double max) {
    return min + (max - min) * rand() / (RAND_MAX + 1.0);
}

// change matrix to array (2D matrix can not be passed to GPU)
std::vector<double> matrix_to_array(const std::vector<std::vector<double>> &matrix) {  
  std::vector<double> array;  
  for (const auto &row : matrix) {  
    array.insert(array.end(), row.begin(), row.end());  
  }  
  return array;  
}  

// main CUDA kernel
__global__ void train_mlp_cuda(Device_MLP_CUDA& mlp_cuda,  double* input, double* labels, double lr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Forward pass
    mlp_cuda.forward(input, idx);

    // clear gradients
    mlp_cuda.zero_grad();

    // Backward pass
    mlp_cuda.backward(labels, input, idx);

    // Update weights and biases
    mlp_cuda.update(lr, idx);
}

// definition of Host MLP class
Host_MLP_CUDA::Host_MLP_CUDA(int input_dim, int hidden_dim, int output_dim){
    // Randomly initialize the weights and biases (the same as CPU)
    W1 = vector<vector<double>>(hidden_dim, vector<double>(input_dim, 0));
    W2 = vector<vector<double>>(output_dim, vector<double>(hidden_dim, 0));
    b1 = vector<double>(hidden_dim, 0);
    b2 = vector<double>(output_dim, 0);

    // initialize W1, W2, b1, b2
    for(int i = 0;i<hidden_dim;i++){
        for (int j = 0; j < input_dim; ++j) {
            W1[i][j] = cu_random(-1,1);
        }
    }
    for(int i = 0;i<output_dim;i++){
        for (int j = 0; j < hidden_dim; ++j) {
            W2[i][j] = cu_random(-1,1);
        }
    }
    for(int i = 0; i<hidden_dim; i++) {
        b1[i] = cu_random(-1,1);
    }
    for (int i = 0; i < output_dim; ++i) {
        b2[i] = cu_random(-1,1);
    }

    // Initialize the gradients
    W1_grad = vector<vector<double>>(W1.size(), vector<double>(W1[0].size(), 0));
    W2_grad = vector<vector<double>>(W2.size(), vector<double>(W2[0].size(), 0));
    b1_grad = vector<double>(b1.size(), 0);
    b2_grad = vector<double>(b2.size(), 0);

    // inner variables should also be prepared to copy to GPU
    y1 = vector<double>(hidden_dim, 0);
    z1 = vector<double>(hidden_dim, 0);
    y2 = vector<double>(output_dim, 0);
    z2 = vector<double>(output_dim, 0);

    // need to transfer 2D array to 1D array before copying to GPU
    W1_1D = matrix_to_array(W1);
    W2_1D = matrix_to_array(W2);
    W1_grad_1D = matrix_to_array(W1_grad);
    W2_grad_1D = matrix_to_array(W2_grad);
}

Host_MLP_CUDA::~Host_MLP_CUDA() = default;

// definition of Device MLP class
Device_MLP_CUDA::Device_MLP_CUDA(int input_dim, int hidden_dim, int output_dim) {
    this->input_dim = input_dim;  
    this->hidden_dim = hidden_dim;  
    this->output_dim = output_dim;
}

void copyDataToDevice(Host_MLP_CUDA& host_mlp, Device_MLP_CUDA& device_mlp){
    // copy the weights and gradients to GPU
    cudaMalloc((void**)&device_mlp.d_W1, host_mlp.W1_1D.size() * sizeof(double));   // should use host_mlp.W1_1D.size() instead of host_mlp.W1.size(). The second one is the number of rows, because W1 is a 2D array.
    cudaMalloc((void**)&device_mlp.d_W2, host_mlp.W2_1D.size() * sizeof(double));   // 2D to 1D
    cudaMalloc((void**)&device_mlp.d_b1, host_mlp.b1.size() * sizeof(double));
    cudaMalloc((void**)&device_mlp.d_b2, host_mlp.b2.size() * sizeof(double));
    cudaMalloc((void**)&device_mlp.d_W1_grad, host_mlp.W1_grad_1D.size() * sizeof(double));  // 2D to 1D
    cudaMalloc((void**)&device_mlp.d_W2_grad, host_mlp.W2_grad_1D.size() * sizeof(double)); // 2D to 1D
    cudaMalloc((void**)&device_mlp.d_b1_grad, host_mlp.b1_grad.size() * sizeof(double));
    cudaMalloc((void**)&device_mlp.d_b2_grad, host_mlp.b2_grad.size() * sizeof(double));
    
    cudaMalloc((void**)&device_mlp.d_y1, host_mlp.y1.size() * sizeof(double));
    cudaMalloc((void**)&device_mlp.d_z1, host_mlp.z1.size() * sizeof(double));
    cudaMalloc((void**)&device_mlp.d_y2, host_mlp.y2.size() * sizeof(double));
    cudaMalloc((void**)&device_mlp.d_z2, host_mlp.z2.size() * sizeof(double));

    // need to transfer 2D array to 1D array before copying to GPU
    cudaMemcpy(device_mlp.d_W1, host_mlp.W1_1D.data(), host_mlp.W1_1D.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_mlp.d_W2, host_mlp.W2_1D.data(), host_mlp.W2_1D.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_mlp.d_W1_grad, host_mlp.W1_grad_1D.data(), host_mlp.W1_grad_1D.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_mlp.d_W2_grad, host_mlp.W2_grad_1D.data(), host_mlp.W2_grad_1D.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_mlp.d_b1, host_mlp.b1.data(), host_mlp.b1.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_mlp.d_b2, host_mlp.b2.data(), host_mlp.b2.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_mlp.d_b1_grad, host_mlp.b1_grad.data(), host_mlp.b1_grad.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_mlp.d_b2_grad, host_mlp.b2_grad.data(), host_mlp.b2_grad.size() * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(device_mlp.d_y1, host_mlp.y1.data(), host_mlp.y1.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_mlp.d_z1, host_mlp.z1.data(), host_mlp.z1.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_mlp.d_y2, host_mlp.y2.data(), host_mlp.y2.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_mlp.d_z2, host_mlp.z2.data(), host_mlp.z2.size() * sizeof(double), cudaMemcpyHostToDevice);

}

__device__ void Device_MLP_CUDA::zero_grad() {
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
// forward CUDA function
__device__ void Device_MLP_CUDA::forward(double* input, int idx) {

    // input -> first hidden layer
    if (idx < hidden_dim) {
        double sum = 0;
        for (int j = 0; j < input_dim; ++j) {
            sum += d_W1[idx * input_dim + j] * input[j];  // matrix-vector multiplication
        }
        d_y1[idx] = sum + d_b1[idx];        // vector add
        d_z1[idx] = cu_sigmoid(d_y1[idx]);     // activation function (sigmoid)
    }
    // first hidden layer -> output
    double out_sum = 0;
    if (idx < output_dim) {
        double sum = 0;
        for (int j = 0; j < hidden_dim; ++j) {
            sum += d_W2[idx * hidden_dim + j] * d_z1[j];    // matrix-vector multiplication
        }
        d_y2[idx] = sum + d_b2[idx];            // vector add
        d_z2[idx] = exp(d_y2[idx]);             // activation function
    }
    __syncthreads();
    // Calculate the sum of d_z2 elements using parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (idx < stride && idx + stride < output_dim) {
            d_z2[idx] += d_z2[idx + stride];
        }
        __syncthreads();
    }

    // Store the sum in out_sum
    if (idx == 0) {
        out_sum = d_z2[0];
    }
    __syncthreads();
    if (idx < output_dim) {
        for (int i = 0; i < output_dim; ++i) {
            d_z2[idx] /= out_sum;               // finish softmax
        }
    }
}


// Backward CUDA function
__device__ void Device_MLP_CUDA::backward(double* y_label, double* input, int idx) {
    // int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_dim) {
        d_b2_grad[idx] = cu_d_softmax_cross_entropy(y_label[idx], d_z2[idx]);      // softmax cross entropy gradient
        for (int j = 0; j < hidden_dim; ++j) {
            d_W2_grad[idx * hidden_dim + j] = d_b2_grad[idx] * d_z1[j];       // outer product
        }
    }
    if (idx < hidden_dim) {
        double sum = 0;
        for (int i = 0; i < output_dim; ++i) {
            sum += d_W2[i * hidden_dim + idx] * d_b2_grad[i];  //! W2 is transposed, and then matrix multiplication with b2_grad
        }
        d_b1_grad[idx] = sum * cu_d_sigmoid(d_y1[idx]);        // sigmoid gradient of y1, and then vector multiplication
        for (int j = 0; j < input_dim; ++j) {
            d_W1_grad[idx * input_dim + j] = d_b1_grad[idx] * input[j];   // outer product, need to use input data
        }
    }
}


__device__ void Device_MLP_CUDA::update(double lr, int idx) {
    // int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_dim * input_dim) {
        d_W1[idx] -= lr * d_W1_grad[idx];   // update W1
    }
    if (idx < output_dim * hidden_dim) {
        d_W2[idx] -= lr * d_W2_grad[idx];   // update W2
    }
    if (idx < hidden_dim) {
        d_b1[idx] -= lr * d_b1_grad[idx];   // update b1
    }
    if (idx < output_dim) {
        d_b2[idx] -= lr * d_b2_grad[idx];   // update b2
    }
}

Device_MLP_CUDA::~Device_MLP_CUDA(){
    cudaFree(d_W1);
    cudaFree(d_W2);
    cudaFree(d_b1);
    cudaFree(d_b2);
    cudaFree(d_W1_grad);
    cudaFree(d_W2_grad);
    cudaFree(d_b1_grad);
    cudaFree(d_b2_grad);
    cudaFree(d_y1);
    cudaFree(d_z1);
    cudaFree(d_y2);
    cudaFree(d_z2);
}

// this function is used in host machine. this step is necessary.
void freeDeviceMemory(Device_MLP_CUDA& device_mlp){
    if (device_mlp.d_W1 != nullptr) cudaFree(device_mlp.d_W1);
    if (device_mlp.d_W2 != nullptr) cudaFree(device_mlp.d_W2);
    if (device_mlp.d_b1 != nullptr) cudaFree(device_mlp.d_b1);
    if (device_mlp.d_b2 != nullptr) cudaFree(device_mlp.d_b2);
    if (device_mlp.d_W1_grad != nullptr) cudaFree(device_mlp.d_W1_grad);
    if (device_mlp.d_W2_grad != nullptr) cudaFree(device_mlp.d_W2_grad);
    if (device_mlp.d_b1_grad != nullptr) cudaFree(device_mlp.d_b1_grad);
    if (device_mlp.d_b2_grad != nullptr) cudaFree(device_mlp.d_b2_grad);
    if (device_mlp.d_y1 != nullptr) cudaFree(device_mlp.d_y1);
    if (device_mlp.d_z1 != nullptr) cudaFree(device_mlp.d_z1);
    if (device_mlp.d_y2 != nullptr) cudaFree(device_mlp.d_y2);
    if (device_mlp.d_z2 != nullptr) cudaFree(device_mlp.d_z2);
}
