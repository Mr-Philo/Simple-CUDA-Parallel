#ifndef CPP_TEMPLATE_MLP_H
#define CPP_TEMPLATE_MLP_H

#include "functions.h"
#include "vector"

class MLP {
public:
    MLP(int input_dim, int hidden_dim, int output_dim);

    std::vector<double> forward(const std::vector<unsigned char> &);

    void zero_grad();

    void backward(const std::vector<double> &y, const std::vector<double> &y_hat);

    void update(double lr);

    ~MLP();

private:
    std::vector<std::vector<double>> W1;
    std::vector<std::vector<double>> W2;
    std::vector<double> b1;
    std::vector<double> b2;
    std::vector<std::vector<double>> W1_grad;
    std::vector<std::vector<double>> W2_grad;
    std::vector<double> b1_grad;
    std::vector<double> b2_grad;
    std::vector<double> input;
    std::vector<double> y1;
    std::vector<double> z1;
};


#endif //CPP_TEMPLATE_MLP_H
