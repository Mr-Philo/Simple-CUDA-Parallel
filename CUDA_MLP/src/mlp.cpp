#include "mlp.h"
using namespace std;

MLP::MLP(int input_dim, int hidden_dim, int output_dim) {
    // Randomly initialize the weights and biases
    W1 = vector<vector<double>>(hidden_dim, vector<double>(input_dim, 0));
    W2 = vector<vector<double>>(output_dim, vector<double>(hidden_dim, 0));
    b1 = vector<double>(hidden_dim, 0);
    b2 = vector<double>(output_dim, 0);

    // initialize W1, W2, b1, b2
    for(int i = 0;i<hidden_dim;i++){
        for (int j = 0; j < input_dim; ++j) {
            W1[i][j] = random(-1,1);
        }
    }
    for(int i = 0;i<output_dim;i++){
        for (int j = 0; j < hidden_dim; ++j) {
            W2[i][j] = random(-1,1);
        }
    }
    for(int i = 0; i<hidden_dim; i++) {
        b1[i] = random(-1,1);
    }
    for (int i = 0; i < output_dim; ++i) {
        b2[i] = random(-1,1);
    }

    // Initialize the gradients
    W1_grad = vector<vector<double>>(W1.size(), vector<double>(W1[0].size(), 0));
    W2_grad = vector<vector<double>>(W2.size(), vector<double>(W2[0].size(), 0));
    b1_grad = vector<double>(b1.size(), 0);
    b2_grad = vector<double>(b2.size(), 0);
}

void MLP::zero_grad() {
    for (auto &i: W1_grad) {
        fill(i.begin(), i.end(), 0);
    }
    for (auto &i: W2_grad) {
        fill(i.begin(), i.end(), 0);
    }
    fill(b1_grad.begin(), b1_grad.end(), 0);
    fill(b2_grad.begin(), b2_grad.end(), 0);
}


vector<double> MLP::forward(const vector<unsigned char> &x) {
    input = vector<double>(x.begin(),x.end());
    y1 = matrix_dot(W1,input);
    vector_add(y1, b1);
    z1 = sigmoid(y1);
    vector<double> y2 = matrix_dot(W2,z1);
    vector_add(y2,b2);
    vector<double> z2 = softmax(y2);
    return z2;
}

void MLP::backward(const vector<double> &y, const vector<double> &y_hat) {
    b2_grad = d_softmax_cross_entropy(y,y_hat);
    W2_grad = outer_product(b2_grad,z1);
    b1_grad = vector_dot(matrix_dot(transpose(W2),b2_grad), d_sigmoid(y1));
    W1_grad = outer_product(b1_grad,input);
}

void MLP::update(double lr) {
    // Update weights and biases for input layer to hidden layer
    matrix_multiply(W1_grad,-lr);
    matrix_add(W1, W1_grad);
    vector_multiply(b1_grad,-lr);
    vector_add(b1,b1_grad);
    matrix_multiply(W2_grad,-lr);
    matrix_add(W2, W2_grad);
    vector_multiply(b2_grad,-lr);
    vector_add(b2,b2_grad);
}

MLP::~MLP() = default;
