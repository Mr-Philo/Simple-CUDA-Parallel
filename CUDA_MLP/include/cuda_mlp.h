#ifndef CUDA_MLP_H_
#define CUDA_MLP_H_

#include <vector>
#include <cuda_functions.h>

class MLP_CUDA {
public:
    MLP_CUDA(int input_dim, int hidden_dim, int output_dim);

    void forward(const std::vector<unsigned char> &);

    void zero_grad();

    void backward(const std::vector<double> &y, const std::vector<double> &y_hat);

    void update(double lr);

    ~MLP_CUDA();

private:
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
    double* d_input;
    double y1;
    double z1;
};

#endif //CUDA_MLP_H_
