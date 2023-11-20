#ifndef CPP_TEMPLATE_FUNCTIONS_H
#define CPP_TEMPLATE_FUNCTIONS_H

#include <vector>
using namespace std;
double cross_entropy(const std::vector<double> &y, const std::vector<double> &y_hat);
std::vector<double> sigmoid(vector<double> x);
std::vector<double> d_sigmoid(const std::vector<double> &x);
std::vector<double> softmax(const std::vector<double> &x);
std::vector<double> d_softmax_cross_entropy(const std::vector<double> &y, const std::vector<double> &y_hat);
std::vector<std::vector<double>> outer_product(const std::vector<double> &x, const std::vector<double> &y);
void matrix_add(std::vector<std::vector<double>> &x, const std::vector<std::vector<double>> &y);
void vector_add(std::vector<double> &x, const std::vector<double> &y);
void matrix_multiply(std::vector<std::vector<double>> &x, double y);
void vector_multiply(std::vector<double> &x, double y);
double random(double min, double max);
vector<double> matrix_dot(vector<vector<double>> a, vector<double> b);
vector<double> vector_dot(vector<double>x, vector<double>y);
vector<vector<double>> transpose(vector<vector<double>> x);
#endif //CPP_TEMPLATE_FUNCTIONS_H
