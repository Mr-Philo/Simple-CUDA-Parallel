#ifndef CUDA_MLP_H_
#define CUDA_MLP_H_

#include <vector>

class Host_MLP_CUDA {
public:
    Host_MLP_CUDA(int input_dim, int hidden_dim, int output_dim);

    // weights and bias
    std::vector<std::vector<double>> W1;
    std::vector<std::vector<double>> W2;
    std::vector<double> b1;
    std::vector<double> b2;
    std::vector<std::vector<double>> W1_grad;
    std::vector<std::vector<double>> W2_grad;
    std::vector<double> b1_grad;
    std::vector<double> b2_grad;

    // inner variables
    std::vector<double> y1;
    std::vector<double> z1;
    std::vector<double> y2;
    std::vector<double> z2;

    // 1D array for CUDA
    std::vector<double> W1_1D;
    std::vector<double> W2_1D;
    std::vector<double> W1_grad_1D;
    std::vector<double> W2_grad_1D;

    ~Host_MLP_CUDA();
};

class Device_MLP_CUDA {
public:
    Device_MLP_CUDA(int input_dim, int hidden_dim, int output_dim);

    __device__ void forward(double* input, int idx);

    __device__ void zero_grad();

    __device__ void backward(double* y_label, double* input, int idx);

    __device__ void update(double lr, int idx);

    ~Device_MLP_CUDA();

    int input_dim;
    int hidden_dim;
    int output_dim;
    double* d_W1;   // 2D
    double* d_W2;   // 2D
    double* d_b1;   
    double* d_b2;
    double* d_W1_grad;    // 2D
    double* d_W2_grad;    // 2D
    double* d_b1_grad;
    double* d_b2_grad;
    double* d_y1;
    double* d_z1;
    double* d_y2;
    double* d_z2;
};

__global__ void train_mlp_cuda(Device_MLP_CUDA& mlp_cuda,  double* input, double* labels, double lr);

void copyDataToDevice(Host_MLP_CUDA& host_mlp, Device_MLP_CUDA& device_mlp);
void freeDeviceMemory(Device_MLP_CUDA& device_mlp);

__device__ double sigmoid(double x);
__device__ double d_sigmoid(double x);
__device__ double softmax(double x);
__device__ double d_softmax_cross_entropy(double y, double y_hat);

double random(double min, double max);
std::vector<double> matrix_to_array(const std::vector<std::vector<double>> &matrix);

#endif //CUDA_MLP_H_
