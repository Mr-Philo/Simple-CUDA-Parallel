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

// main CUDA kernel
__global__ void train_mlp_cuda(MLP_CUDA* mlp_cuda,  double* input, double* labels, double lr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0) {
        printf("\n-----------------------precheck-----------------------------------\n");
        printf("input dim: %i\n", mlp_cuda->input_dim);
        printf("hidden dim: %i\n", mlp_cuda->hidden_dim);
        printf("output dim: %i\n", mlp_cuda->output_dim);
        printf("weight W1: \n");
        for (int i = 0; i < 10; ++i) {printf("%f ", mlp_cuda->W1[i]);}  // only print the first 10 elements
        printf("\ngrad : \n");
        for (int i = 0; i < 10; ++i) {printf("%f ", mlp_cuda->W1_grad[i]);}
    }

    // Forward pass
    forward(mlp_cuda, input, idx);
    __syncthreads();
    if (idx == 0) {
        printf("\n-----------------------forward-----------------------------------\n");
        printf("check output: ");
        for (int i = 0; i < mlp_cuda->output_dim; ++i) {
            printf("%f ", mlp_cuda->z2[i]);
        }
    }
    // clear gradients
    zero_grad(mlp_cuda);
    // __syncthreads();
    // if (idx == 0) {
    //     printf("\n-----------------------zero_grad-----------------------------------\n");
    //     printf("check if gradients are cleared: \n");
    //     for (int i = 0; i < mlp_cuda->hidden_dim * mlp_cuda->input_dim; ++i) {printf("%f ", mlp_cuda->W1_grad[i]);}
    // }

    // Backward pass
    backward(mlp_cuda, labels, input, idx);
    __syncthreads();
    // if (idx == 0) {
        // printf("\n-----------------------backward-----------------------------------\n");
    //     printf("check gradients: \n");
    //     for (int i = 0; i < mlp_cuda->hidden_dim * mlp_cuda->input_dim; ++i) {printf("%f ", mlp_cuda->W1_grad[i]);}
    // }

    // Update weights and biases
    update(mlp_cuda, lr, idx);

    __syncthreads();
    if (idx == 0) {
        printf("\n--------------------------final-----------------------------------------\n");
        printf("check label: \n");
        // printf("%f ", labels[0]);
        // printf("%i ", mlp_cuda->output_dim);    // previous error: output_dim is 0
        for (int i = 0; i < mlp_cuda->output_dim; ++i) {printf("%f ", labels[i]);}
        printf("check output: \n");
        for (int i = 0; i < mlp_cuda->output_dim; ++i) {printf("%f ", mlp_cuda->z2[i]);}
    }
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

    // Initialize the gradients
    mlp_cuda->W1_grad = new double[hidden_dim * input_dim];
    mlp_cuda->W2_grad = new double[output_dim * hidden_dim];
    mlp_cuda->b1_grad = new double[hidden_dim];
    mlp_cuda->b2_grad = new double[output_dim];

    // inner variables should also be prepared to copy to GPU
    mlp_cuda->y1 = new double[hidden_dim];
    mlp_cuda->z1 = new double[hidden_dim];
    mlp_cuda->y2 = new double[output_dim];
    mlp_cuda->z2 = new double[output_dim];
    
}


// init Device MLP
void Init_Device_MLP(MLP_CUDA* h_mlp_cuda, MLP_CUDA** d_mlp_cuda){
    /*首先，我们需要理解指针是什么。指针是一个变量，其值为另一个变量的地址，即直接指向另一个变量。当我们声明MLP_CUDA** d_mlp_cuda时，我们创建了一个指针d_mlp_cuda，它指向一个MLP_CUDA类型的变量。

    然后，为什么我们需要一个指向指针的指针MLP_CUDA** d_mlp_cuda呢？这是因为我们在这个函数中要改变指针d_mlp_cuda所指向的地址。在C++中，函数参数是通过值传递的，也就是说，当我们把一个变量传递给一个函数时，函数会创建这个变量的一个副本。因此，如果我们只传递一个指针MLP_CUDA* d_mlp_cuda给函数，函数会得到这个指针的副本，对这个副本的修改不会影响到原来的指针。但是，如果我们传递一个指向指针的指针MLP_CUDA** d_mlp_cuda给函数，函数就可以通过这个指向指针的指针修改原来的指针，使其指向一个新的地址。*/
    int input_dim = h_mlp_cuda->input_dim;
    int hidden_dim = h_mlp_cuda->hidden_dim;
    int output_dim = h_mlp_cuda->output_dim;

    // allocate memory for the pointer to the device MLP pointer (double pointer)
    cudaMalloc((void**)d_mlp_cuda, sizeof(MLP_CUDA));

    // Copy the integer variables input_dim, hidden_dim, output_dim to the device MLP structure
    cudaMemcpy(&(*d_mlp_cuda)->input_dim, &h_mlp_cuda->input_dim, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&(*d_mlp_cuda)->hidden_dim, &h_mlp_cuda->hidden_dim, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&(*d_mlp_cuda)->output_dim, &h_mlp_cuda->output_dim, sizeof(int), cudaMemcpyHostToDevice);

    // allocate memory for the double array in the device MLP
    double *d_W1, *d_W2, *d_b1, *d_b2, *d_W1_grad, *d_W2_grad, *d_b1_grad, *d_b2_grad, *d_y1, *d_z1, *d_y2, *d_z2;
    cudaMalloc((void**)&d_W1, hidden_dim * input_dim * sizeof(double));
    cudaMalloc((void**)&d_W2, output_dim * hidden_dim * sizeof(double));
    cudaMalloc((void**)&d_b1, hidden_dim * sizeof(double));
    cudaMalloc((void**)&d_b2, output_dim * sizeof(double));
    cudaMalloc((void**)&d_W1_grad, hidden_dim * input_dim * sizeof(double));
    cudaMalloc((void**)&d_W2_grad, output_dim * hidden_dim * sizeof(double));
    cudaMalloc((void**)&d_b1_grad, hidden_dim * sizeof(double));
    cudaMalloc((void**)&d_b2_grad, output_dim * sizeof(double));
    cudaMalloc((void**)&d_y1, hidden_dim * sizeof(double));
    cudaMalloc((void**)&d_z1, hidden_dim * sizeof(double));
    cudaMalloc((void**)&d_y2, output_dim * sizeof(double));
    cudaMalloc((void**)&d_z2, output_dim * sizeof(double));
    
    // copy the array data from host to device
    cudaMemcpy(d_W1, h_mlp_cuda->W1, hidden_dim * input_dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, h_mlp_cuda->W2, output_dim * hidden_dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, h_mlp_cuda->b1, hidden_dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, h_mlp_cuda->b2, output_dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1_grad, h_mlp_cuda->W1_grad, hidden_dim * input_dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2_grad, h_mlp_cuda->W2_grad, output_dim * hidden_dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1_grad, h_mlp_cuda->b1_grad, hidden_dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2_grad, h_mlp_cuda->b2_grad, output_dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y1, h_mlp_cuda->y1, hidden_dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z1, h_mlp_cuda->z1, hidden_dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y2, h_mlp_cuda->y2, output_dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z2, h_mlp_cuda->z2, output_dim * sizeof(double), cudaMemcpyHostToDevice);

    // copy the array pointer to the device MLP_struct pointer (all the pointers are in the device)
    cudaMemcpy(&((*d_mlp_cuda)->W1), &d_W1, sizeof(double*), cudaMemcpyHostToDevice);
    /*这行代码的目的是将主机（CPU）上的指针d_W1的值（也就是设备（GPU）上的double数组的地址）复制到设备上的结构体成员d_struct->d_W1中。

    cudaMemcpy函数的原型为cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)，它用于将数据从源地址src复制到目标地址dst。

    在这行代码中：
    &((*d_struct)->d_W1)是目标地址。(*d_struct)->d_W1访问设备上的结构体的d_W1成员，&操作符获取这个成员的地址。因此，&((*d_struct)->d_W1)得到的是设备上的结构体成员d_W1的地址。
    &d_W1是源地址。d_W1是主机上的一个指针，它的值是设备上的double数组的地址，&操作符获取这个指针的地址。因此，&d_W1得到的是主机上的指针d_W1的地址，这个指针的值是设备上的double数组的地址。
    sizeof(double*)是复制的数据量，因为我们要复制一个指针，所以数据量为一个指针的大小。
    cudaMemcpyHostToDevice表示数据的复制方向是从主机到设备。*/
    cudaMemcpy(&((*d_mlp_cuda)->W2), &d_W2, sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(&((*d_mlp_cuda)->b1), &d_b1, sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(&((*d_mlp_cuda)->b2), &d_b2, sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(&((*d_mlp_cuda)->W1_grad), &d_W1_grad, sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(&((*d_mlp_cuda)->W2_grad), &d_W2_grad, sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(&((*d_mlp_cuda)->b1_grad), &d_b1_grad, sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(&((*d_mlp_cuda)->b2_grad), &d_b2_grad, sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(&((*d_mlp_cuda)->y1), &d_y1, sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(&((*d_mlp_cuda)->z1), &d_z1, sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(&((*d_mlp_cuda)->y2), &d_y2, sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(&((*d_mlp_cuda)->z2), &d_z2, sizeof(double*), cudaMemcpyHostToDevice);

}

void Copy_Device_to_Host(MLP_CUDA* h_mlp_cuda, MLP_CUDA* d_mlp_cuda){
    double *d_W1, *d_W2, *d_b1, *d_b2, *d_W1_grad, *d_W2_grad, *d_b1_grad, *d_b2_grad, *d_y1, *d_z1, *d_y2, *d_z2;
    // get the array pointer from the device MLP pointer
    cudaMemcpy(&d_W1, &(d_mlp_cuda->W1), sizeof(double*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_W2, &(d_mlp_cuda->W2), sizeof(double*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_b1, &(d_mlp_cuda->b1), sizeof(double*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_b2, &(d_mlp_cuda->b2), sizeof(double*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_W1_grad, &(d_mlp_cuda->W1_grad), sizeof(double*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_W2_grad, &(d_mlp_cuda->W2_grad), sizeof(double*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_b1_grad, &(d_mlp_cuda->b1_grad), sizeof(double*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_b2_grad, &(d_mlp_cuda->b2_grad), sizeof(double*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_y1, &(d_mlp_cuda->y1), sizeof(double*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_z1, &(d_mlp_cuda->z1), sizeof(double*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_y2, &(d_mlp_cuda->y2), sizeof(double*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_z2, &(d_mlp_cuda->z2), sizeof(double*), cudaMemcpyDeviceToHost);

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


__device__ void setZero(double* arr, int size) {  
    int idx = threadIdx.x;  
    if (idx < size) {  
        arr[idx] = 0.0;  
    }  
} 


// zero_grad CUDA function
__device__ void zero_grad(MLP_CUDA* d_mlp_cuda){

    int input_dim = d_mlp_cuda->input_dim;
    int hidden_dim = d_mlp_cuda->hidden_dim;
    int output_dim = d_mlp_cuda->output_dim;

    // fill the gradients with 0
    setZero(d_mlp_cuda->W1_grad, hidden_dim * input_dim);
    setZero(d_mlp_cuda->W2_grad, output_dim * hidden_dim);
    setZero(d_mlp_cuda->b1_grad, hidden_dim);
    setZero(d_mlp_cuda->b2_grad, output_dim);
}


// TODO: 更新：将CUDA内核缩减至一个，只在main.cpp函数中执行，其中整个流程的主要函数采取_device__的方式，即在CUDA内核中执行
// forward CUDA function
__device__ void forward(MLP_CUDA* d_mlp_cuda, double* input, int idx) {

    int input_dim = d_mlp_cuda->input_dim;
    int hidden_dim = d_mlp_cuda->hidden_dim;
    int output_dim = d_mlp_cuda->output_dim;

    // input -> first hidden layer
    if (idx < hidden_dim) {
        double sum = 0;
        for (int j = 0; j < input_dim; ++j) {
            sum += d_mlp_cuda->W1[idx * input_dim + j] * input[j];  // matrix-vector multiplication
        }
        d_mlp_cuda->y1[idx] = sum + d_mlp_cuda->b1[idx];        // vector add
        d_mlp_cuda->z1[idx] = cu_sigmoid(d_mlp_cuda->y1[idx]);     // activation function (sigmoid)
    }
    // first hidden layer -> output
    double out_sum = 0;
    if (idx < output_dim) {
        double sum = 0;
        for (int j = 0; j < hidden_dim; ++j) {
            sum += d_mlp_cuda->W2[idx * hidden_dim + j] * d_mlp_cuda->z1[j];    // matrix-vector multiplication
        }
        d_mlp_cuda->y2[idx] = sum + d_mlp_cuda->b2[idx];            // vector add
        d_mlp_cuda->z2[idx] = cu_softmax(d_mlp_cuda->y2[idx]);             // activation function
    }
    __syncthreads();
    // Calculate the sum of d_z2 elements using parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (idx < stride && idx + stride < output_dim) {
            d_mlp_cuda->z2[idx] += d_mlp_cuda->z2[idx + stride];
        }
        __syncthreads();
    }

    // Store the sum in out_sum
    if (idx == 0) {
        out_sum = d_mlp_cuda->z2[0];
    }
    __syncthreads();
    if (idx < output_dim) {
        for (int i = 0; i < output_dim; ++i) {
            d_mlp_cuda->z2[idx] = d_mlp_cuda->z2[idx] / out_sum;               // finish softmax
        }
    }
}


// Backward CUDA function
__device__ void backward(MLP_CUDA* d_mlp_cuda, double* y_label, double* input, int idx) {

    int input_dim = d_mlp_cuda->input_dim;
    int hidden_dim = d_mlp_cuda->hidden_dim;
    int output_dim = d_mlp_cuda->output_dim;

    // int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_dim) {
        d_mlp_cuda->b2_grad[idx] = cu_d_softmax_cross_entropy(d_mlp_cuda->z2[idx], y_label[idx]);      // softmax cross entropy gradient
        for (int j = 0; j < hidden_dim; ++j) {
            d_mlp_cuda->W2_grad[idx * hidden_dim + j] = d_mlp_cuda->b2_grad[idx] * d_mlp_cuda->z1[j];       // outer product
        }
    }
    if (idx < hidden_dim) {
        double sum = 0;
        for (int i = 0; i < output_dim; ++i) {
            // sum += d_W2[i * hidden_dim + idx] * d_b2_grad[i];  //! W2 is transposed, and then matrix multiplication with b2_grad
            sum += d_mlp_cuda->W2[idx * output_dim + i] * d_mlp_cuda->b2_grad[i];  //! W2 is transposed, and then matrix multiplication with b2_grad
        }
        d_mlp_cuda->b1_grad[idx] = sum * cu_d_sigmoid(d_mlp_cuda->y1[idx]);        // sigmoid gradient of y1, and then vector multiplication
        for (int j = 0; j < input_dim; ++j) {
            d_mlp_cuda->W1_grad[idx * input_dim + j] = d_mlp_cuda->b1_grad[idx] * input[j];   // outer product, need to use input data
        }
    }
}


__device__ void update(MLP_CUDA* d_mlp_cuda, double lr, int idx) {

    int input_dim = d_mlp_cuda->input_dim;
    int hidden_dim = d_mlp_cuda->hidden_dim;
    int output_dim = d_mlp_cuda->output_dim;

    // int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_dim * input_dim) {
        d_mlp_cuda->W1[idx] -= lr * d_mlp_cuda->W1_grad[idx];   // update W1
    }
    if (idx < output_dim * hidden_dim) {
        d_mlp_cuda->W2[idx] -= lr * d_mlp_cuda->W2_grad[idx];   // update W2
    }
    if (idx < hidden_dim) {
        d_mlp_cuda->b1[idx] -= lr * d_mlp_cuda->b1_grad[idx];   // update b1
    }
    if (idx < output_dim) {
        d_mlp_cuda->b2[idx] -= lr * d_mlp_cuda->b2_grad[idx];   // update b2
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

void Free_Device_MLP(MLP_CUDA* d_mlp_cuda){
    double *d_W1, *d_W2, *d_b1, *d_b2, *d_W1_grad, *d_W2_grad, *d_b1_grad, *d_b2_grad, *d_y1, *d_z1, *d_y2, *d_z2;
    // get the array pointer from the device MLP pointer
    cudaMemcpy(&d_W1, &(d_mlp_cuda->W1), sizeof(double*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_W2, &(d_mlp_cuda->W2), sizeof(double*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_b1, &(d_mlp_cuda->b1), sizeof(double*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_b2, &(d_mlp_cuda->b2), sizeof(double*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_W1_grad, &(d_mlp_cuda->W1_grad), sizeof(double*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_W2_grad, &(d_mlp_cuda->W2_grad), sizeof(double*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_b1_grad, &(d_mlp_cuda->b1_grad), sizeof(double*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_b2_grad, &(d_mlp_cuda->b2_grad), sizeof(double*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_y1, &(d_mlp_cuda->y1), sizeof(double*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_z1, &(d_mlp_cuda->z1), sizeof(double*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_y2, &(d_mlp_cuda->y2), sizeof(double*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d_z2, &(d_mlp_cuda->z2), sizeof(double*), cudaMemcpyDeviceToHost);

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
    cudaFree(d_mlp_cuda);
}
