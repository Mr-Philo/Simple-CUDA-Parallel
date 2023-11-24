#ifndef CUDA_MLP_H_
#define CUDA_MLP_H_

#include <vector>

struct MLP_CUDA {
    int input_dim;
    int hidden_dim;
    int output_dim;
    double* W1;   // 2D
    double* W2;   // 2D
    double* b1;   
    double* b2;
    double* W1_grad;    // 2D
    double* W2_grad;    // 2D
    double* b1_grad;
    double* b2_grad;
    double* y1;
    double* z1;
    double* y2;
    double* z2;
};

// __global__ void train_mlp_cuda(MLP_CUDA* mlp_cuda,  double* input, double* labels, double lr);

void Init_Host_MLP(MLP_CUDA* mlp_cuda, int input_dim, int hidden_dim, int output_dim);
void Init_Device_MLP(MLP_CUDA* h_mlp_cuda, MLP_CUDA** d_mlp_cuda);
void Init_Device_MLP_NEW(MLP_CUDA* h_mlp_cuda, MLP_CUDA* d_mlp_cuda);
void Copy_Device_to_Host(MLP_CUDA* h_mlp_cuda, MLP_CUDA* d_mlp_cuda);
void Free_Host_MLP(MLP_CUDA* mlp_cuda);
void Free_Device_MLP(MLP_CUDA* d_mlp_cuda);

__device__ double cu_sigmoid(double x);
__device__ double cu_d_sigmoid(double x);
__device__ double cu_softmax(double x);
__device__ double cu_d_softmax_cross_entropy(double y, double y_hat);

double cu_random(double min, double max);
std::vector<double> matrix_to_array(const std::vector<std::vector<double>> &matrix);

__global__ void set_zero_matrix_kernel(double* matrix, int row, int col);
__global__ void matrix_vector_mul(double* matrix, double* vector, double* result, int row, int col);
__global__ void matrix_outer_product(double* vector1, double* vector2, double* result, int row, int col);
__global__ void vector_add(double* vector1, double* vector2, double* result, int size);
__global__ void one_layer_forward_sigmoid_kernel(double* input, double* W, double* b, double* y, double* z, int row, int col);
__global__ void one_layer_forward_softmax_kernel(double* input, double* W, double* b, double* y, double* z, int row, int col);
__global__ void one_layer_backward_sigmoid_kernel(double* y, double* W, double* b_grad, double* W_grad, double* b, double* input, int row, int col);
__global__ void one_layer_backward_softmax_kernel(double* y, double* z, double* y_label, double* W_grad, double* b_grad, int row, int col);
__global__ void matrix_update_kernel(double* W, double* W_grad, double lr, int row, int col);

__device__ void setZero(double* arr, int size);
__device__ void zero_grad(MLP_CUDA* d_mlp_cuda);
__device__ void forward(MLP_CUDA* d_mlp_cuda, double* input, int idx);
__device__ void backward(MLP_CUDA* d_mlp_cuda, double* y_label, double* input, int idx);
__device__ void update(MLP_CUDA* d_mlp_cuda, double lr, int idx);

#endif //CUDA_MLP_H_
