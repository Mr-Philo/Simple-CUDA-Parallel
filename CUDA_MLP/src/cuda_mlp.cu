#include "cuda_mlp.h"
#include <cuda_runtime.h>
#include <stdio.h>  

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


// init Host MLP
void Init_Host_MLP(MLP_CUDA* mlp_cuda, int input_dim, int hidden_dim, int output_dim){
    // Randomly initialize the weights and biases
    mlp_cuda->input_dim = input_dim;
    mlp_cuda->hidden_dim = hidden_dim;
    mlp_cuda->output_dim = output_dim;
    mlp_cuda->W1 = new double[hidden_dim * input_dim];
    mlp_cuda->W2 = new double[output_dim * hidden_dim];
    mlp_cuda->b1 = new double[hidden_dim];
    mlp_cuda->b2 = new double[output_dim];

    // initialize W1, W2, b1, b2
    for(int i = 0; i < hidden_dim; i++){
        for (int j = 0; j < input_dim; ++j) {
            mlp_cuda->W1[i * input_dim + j] = cu_random(-1,1);
        }
    }
    for(int i = 0; i < output_dim; i++){
        for (int j = 0; j < hidden_dim; ++j) {
            mlp_cuda->W2[i * hidden_dim + j] = cu_random(-1,1);
        }
    }
    for(int i = 0; i < hidden_dim; i++) {
        mlp_cuda->b1[i] = cu_random(-1,1);
    }
    for (int i = 0; i < output_dim; ++i) {
        mlp_cuda->b2[i] = cu_random(-1,1);
    }

    // Initialize the gradients and set them to zero
    mlp_cuda->W1_grad = new double[hidden_dim * input_dim];
    std::fill(mlp_cuda->W1_grad, mlp_cuda->W1_grad + hidden_dim * input_dim, 0.0);
    mlp_cuda->W2_grad = new double[output_dim * hidden_dim];
    std::fill(mlp_cuda->W2_grad, mlp_cuda->W2_grad + output_dim * hidden_dim, 0.0);
    mlp_cuda->b1_grad = new double[hidden_dim];
    std::fill(mlp_cuda->b1_grad, mlp_cuda->b1_grad + hidden_dim, 0.0);
    mlp_cuda->b2_grad = new double[output_dim];
    std::fill(mlp_cuda->b2_grad, mlp_cuda->b2_grad + output_dim, 0.0);

    // inner variables should also be prepared to copy to GPU. remember to give them initial values
    mlp_cuda->y1 = new double[hidden_dim];
    std::fill(mlp_cuda->y1, mlp_cuda->y1 + hidden_dim, 0.0);
    mlp_cuda->z1 = new double[hidden_dim];
    std::fill(mlp_cuda->z1, mlp_cuda->z1 + hidden_dim, 0.0);
    mlp_cuda->y2 = new double[output_dim];
    std::fill(mlp_cuda->y2, mlp_cuda->y2 + output_dim, 0.0);
    mlp_cuda->z2 = new double[output_dim];
    std::fill(mlp_cuda->z2, mlp_cuda->z2 + output_dim, 0.0);
    
}


// init Device MLP
void Init_Device_MLP(MLP_CUDA* h_mlp_cuda, double** d_W1, double** d_W2, double** d_b1, double** d_b2, double** d_W1_grad, double** d_W2_grad, double** d_b1_grad, double** d_b2_grad, double** d_y1, double** d_z1, double** d_y2, double** d_z2){

    // must be double** d_W1, not double* d_W1
    int input_dim = h_mlp_cuda->input_dim;
    int hidden_dim = h_mlp_cuda->hidden_dim;
    int output_dim = h_mlp_cuda->output_dim;

    cudaMalloc((void**)d_W1, hidden_dim * input_dim * sizeof(double));
    cudaMalloc((void**)d_W2, output_dim * hidden_dim * sizeof(double));
    cudaMalloc((void**)d_b1, hidden_dim * sizeof(double));
    cudaMalloc((void**)d_b2, output_dim * sizeof(double));
    cudaMalloc((void**)d_W1_grad, hidden_dim * input_dim * sizeof(double));
    cudaMalloc((void**)d_W2_grad, output_dim * hidden_dim * sizeof(double));
    cudaMalloc((void**)d_b1_grad, hidden_dim * sizeof(double));
    cudaMalloc((void**)d_b2_grad, output_dim * sizeof(double));
    cudaMalloc((void**)d_y1, hidden_dim * sizeof(double));
    cudaMalloc((void**)d_z1, hidden_dim * sizeof(double));
    cudaMalloc((void**)d_y2, output_dim * sizeof(double));
    cudaMalloc((void**)d_z2, output_dim * sizeof(double));

    // copy the array data from host to device
    cudaMemcpy(*d_W1, h_mlp_cuda->W1, hidden_dim * input_dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_W2, h_mlp_cuda->W2, output_dim * hidden_dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_b1, h_mlp_cuda->b1, hidden_dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_b2, h_mlp_cuda->b2, output_dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_W1_grad, h_mlp_cuda->W1_grad, hidden_dim * input_dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_W2_grad, h_mlp_cuda->W2_grad, output_dim * hidden_dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_b1_grad, h_mlp_cuda->b1_grad, hidden_dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_b2_grad, h_mlp_cuda->b2_grad, output_dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_y1, h_mlp_cuda->y1, hidden_dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_z1, h_mlp_cuda->z1, hidden_dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_y2, h_mlp_cuda->y2, output_dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_z2, h_mlp_cuda->z2, output_dim * sizeof(double), cudaMemcpyHostToDevice);
}

void Copy_Device_to_Host(MLP_CUDA* h_mlp_cuda, double* d_W1, double* d_W2, double* d_b1, double* d_b2, double* d_W1_grad, double* d_W2_grad, double* d_b1_grad, double* d_b2_grad, double* d_y1, double* d_z1, double* d_y2, double* d_z2){

    // copy the array data from device to host
    int input_dim = h_mlp_cuda->input_dim;
    int hidden_dim = h_mlp_cuda->hidden_dim;
    int output_dim = h_mlp_cuda->output_dim;
    cudaMemcpy(h_mlp_cuda->W1, d_W1, hidden_dim * input_dim * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mlp_cuda->W2, d_W2, output_dim * hidden_dim * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mlp_cuda->b1, d_b1, hidden_dim * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mlp_cuda->b2, d_b2, output_dim * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mlp_cuda->W1_grad, d_W1_grad, hidden_dim * input_dim * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mlp_cuda->W2_grad, d_W2_grad, output_dim * hidden_dim * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mlp_cuda->b1_grad, d_b1_grad, hidden_dim * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mlp_cuda->b2_grad, d_b2_grad, output_dim * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mlp_cuda->y1, d_y1, hidden_dim * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mlp_cuda->z1, d_z1, hidden_dim * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mlp_cuda->y2, d_y2, output_dim * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mlp_cuda->z2, d_z2, output_dim * sizeof(double), cudaMemcpyDeviceToHost);

}

__global__ void set_zero_matrix_kernel(double* matrix, int row, int col) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < row * col) {
        matrix[idx] = 0.0;
    }
}

__global__ void matrix_vector_mul(double* matrix, double* vector, double* result, int row, int col) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < row) {
        double sum = 0;
        for (int j = 0; j < col; ++j) {
            sum += matrix[idx * col + j] * vector[j];
        }
        result[idx] = sum;
    }
}

__global__ void matrix_outer_product(double* vector1, double* vector2, double* result, int row, int col) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < row) {
        for (int j = 0; j < col; ++j) {
            result[idx * col + j] = vector1[idx] * vector2[j];
        }
    }
}

__global__ void vector_add(double* vector1, double* vector2, double* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = vector1[idx] + vector2[idx];
    }
}

__global__ void one_layer_forward_sigmoid_kernel(double* input, double* W, double* b, double* y, double* z, int row, int col) {
    // row: input_dim, col: hidden_dim. input should have dim (1, input_dim). result(z) should have dim (1, hidden_dim)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < col) {
        double sum = 0;
        for (int j = 0; j < row; ++j) {
            sum += W[idx * row + j] * input[j];
        }
        y[idx] = sum + b[idx];
        z[idx] = 1 / (1 + exp(-y[idx]));        // sigmoid function
    }
    // if (idx == 0) {     // check whether y is null
    //     printf("----cuda----\n");
    //     printf(input==NULL?"input is null\n":"input is not null\n");
    //     printf(W==NULL?"W is null\n":"W is not null\n");
    //     printf(b==NULL?"b is null\n":"b is not null\n");
    //     printf(y==NULL?"y is null\n":"y is not null\n");
    //     printf(z==NULL?"z is null\n":"z is not null\n");
    // }
    // printf("idx: %d, row: %d, col: %d\n", threadIdx.x, row, col);
}

__global__ void one_layer_forward_softmax_kernel(double* input, double* W, double* b, double* y, double* z, int row, int col) {
    // row: input_dim, col: hidden_dim. input should have dim (1, input_dim). result(z) should have dim (1, hidden_dim)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < col) {
        double sum = 0;
        for (int j = 0; j < row; ++j) {
            sum += W[idx * row + j] * input[j];
        }
        y[idx] = sum + b[idx];
        // z[idx] = cu_softmax(y[idx]);
        z[idx] = exp(y[idx]);        // softmax function
    }
}

__global__ void softmax_normalization_kernel(double* z, int size){
    // z should have dim (1, hidden_dim). no parallelization in this function
    double sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += z[i];
    }
    for (int i = 0; i < size; ++i) {
        z[i] /= sum;
    }
}

__global__ void one_layer_backward_sigmoid_kernel(double* input, double* y_output, double* W_next, double* b_grad_next, double* W_grad, double* b_grad, int row, int col, int next_col) {
    // in back propagation, we need to use the output of the next layer to calculate the gradient of the current layer
    // row: input_dim, col: hidden_dim. next_row and next_col are the dim of the next layer, but here we got next_row = col

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < col) {
        double part_L_to_y = 0;     // part_L_to_y should be a matrix with dim (hidden_dim, 1), the same shape as y
        for (int j = 0; j < next_col; ++j) {        //! need to use 'next_col': the dim of the next layer
            part_L_to_y += W_next[j * col + idx] * b_grad_next[j];      // part_L_to_y = W_next(T) * b_grad_next
        }
        // vector dot product between part_L_to_y and the derivative of sigmoid function
        b_grad[idx] = part_L_to_y * cu_d_sigmoid(y_output[idx]);    // d_sigmoid = sigmoid * (1 - sigmoid)
        for (int j = 0; j < row; ++j) {     // matrix outer product
            W_grad[idx * row + j] = b_grad[idx] * input[j];
        }
    }
}


__global__ void one_layer_backward_softmax_kernel(double* input, double* output, double* y_label, double* W_grad, double* b_grad, int row, int col) {
    // here we already now that this layer is the last layer
    // row: input_dim(hidden_dim here), col: output_dim.

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < col) {
        b_grad[idx] = output[idx] - y_label[idx];    // d_softmax_cross_entropy = y_hat - y
        for (int j = 0; j < row; ++j) {     // matrix outer product
            W_grad[idx * row + j] = b_grad[idx] * input[j];
        }
    }
    // // check whether W_grad is null
    // if (idx == 0) {
    //     printf("----cuda backward----\n");
    //     printf("row: %d, col: %d\n", row, col);
    //     printf(input==NULL?"input is null\n":"input is not null\n");
    //     printf(output==NULL?"output is null\n":"output is not null\n");
    //     printf(y_label==NULL?"y_label is null\n":"y_label is not null\n");
    //     printf(W_grad==NULL?"W_grad is null\n":"W_grad is not null\n");
    //     printf(b_grad==NULL?"b_grad is null\n":"b_grad is not null\n");
    // }
}

__global__ void matrix_update_kernel(double* W, double* W_grad, double lr, int row, int col) {
    // also can be used to update b, just make col = 1
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < row * col) {
        W[idx] -= lr * W_grad[idx];
    }
}

void Free_Host_MLP(MLP_CUDA* mlp_cuda){
    delete[] mlp_cuda->W1;
    delete[] mlp_cuda->W2;
    delete[] mlp_cuda->b1;
    delete[] mlp_cuda->b2;
    delete[] mlp_cuda->W1_grad;
    delete[] mlp_cuda->W2_grad;
    delete[] mlp_cuda->b1_grad;
    delete[] mlp_cuda->b2_grad;
    delete[] mlp_cuda->y1;
    delete[] mlp_cuda->z1;
    delete[] mlp_cuda->y2;
    delete[] mlp_cuda->z2;
}

void Free_Device_MLP(double* d_W1, double* d_W2, double* d_b1, double* d_b2, double* d_W1_grad, double* d_W2_grad, double* d_b1_grad, double* d_b2_grad, double* d_y1, double* d_z1, double* d_y2, double* d_z2){
    // free the array data from device
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
