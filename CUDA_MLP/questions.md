我正在编写一个CUDA项目工程，在cuda_mlp.h中我分别定义了宿主类和设备类如下：
#ifndef CUDA_MLP_H_
#define CUDA_MLP_H_

#include <vector>
#include <cuda_functions.h>

class Host_MLP_CUDA {
public:
    Host_MLP_CUDA(int input_dim, int hidden_dim, int output_dim);
    std::vector<std::vector<double>> W1;
    std::vector<std::vector<double>> W2;
    ......
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
    ......
};

void copyDataToDevice(Host_MLP_CUDA& host_mlp, Device_MLP_CUDA& device_mlp);
void freeDeviceMemory(Device_MLP_CUDA& device_mlp);

#endif //CUDA_MLP_H_
在main.cu中，我先实例化了两个类并将其传入CUDA设备：
Host_MLP_CUDA h_mlp_cuda(784, hidden_dim, 10);
Device_MLP_CUDA d_mlp_cuda(784, hidden_dim, 10);
copyDataToDevice(h_mlp_cuda, d_mlp_cuda);
然后我在主函数中定义了CUDA内核函数：
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
并利用下面语句启动它：
train_mlp_cuda<<<num_of_blocks, block_size>>>(d_mlp_cuda, d_x, d_y, learning_rate);
但在编译main.cu函数时，发生报错：
ptxas fatal   : Unresolved extern function '_ZN15Device_MLP_CUDA7forwardEPdi'
CMakeFiles/mlp.dir/build.make:206: recipe for target 'CMakeFiles/mlp.dir/src/main.cu.o' failed
请帮我分析一下我上述代码的逻辑是否正确，以及错误发生的原因及如何修正它