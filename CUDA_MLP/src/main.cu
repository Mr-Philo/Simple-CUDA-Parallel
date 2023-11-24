#include <unistd.h>
#include <iostream>
#include <string>
#include <getopt.h>
#include <cassert>
#include "utils.h"
#include "mnist_reader_less.h"
#include "mlp.h"
#include "functions.h"
#include "cuda_mlp.h"
// #include <cuda_runtime.h>

using namespace std;

void train(double learning_rate, int epoch_num, int hidden_dim, const string &dataset_path) {
    printf("Learning rate: %f, epoch number: %d, hidden dimension: %d, dataset path: %s\n", learning_rate, epoch_num, hidden_dim, dataset_path.c_str());
    // Read the MNIST dataset
    auto training_images = mnist::read_mnist_image_file<uint8_t>(dataset_path + "/train-images-idx3-ubyte");
    auto training_labels = mnist::read_mnist_label_file<uint8_t>(dataset_path + "/train-labels-idx1-ubyte");
    auto test_images = mnist::read_mnist_image_file<uint8_t>(dataset_path + "/t10k-images-idx3-ubyte");
    auto test_labels = mnist::read_mnist_label_file<uint8_t>(dataset_path + "/t10k-labels-idx1-ubyte");
    printf("Training images: %zu x %zu\n", training_images.size(), training_images[0].size());
    printf("Training labels: %zu\n", training_labels.size());
    assert(training_images.size() == training_labels.size());
    printf("Test images: %zu x %zu\n", test_images.size(), test_images[0].size());
    printf("Test labels: %zu\n", test_labels.size());
    assert(test_images.size() == test_labels.size());

    // Create a neural network with 784 inputs, 100 hidden neurons and 10 outputs
    MLP mlp(784, hidden_dim, 10);

    // Train the network
    for (int epoch = 0; epoch < epoch_num; epoch++) {
        vector<double> losses;
        for (int i = 0; i < training_images.size(); i++) {
            auto x = training_images[i];
            auto l = training_labels[i];
            vector<double> y(10, 0);
            y[l] = 1;
            auto y_hat = mlp.forward(x);
            auto loss = cross_entropy(y, y_hat);
            losses.push_back(loss);
            if (i % 1000 == 0) {
                double sum = 0;
                for (auto &l: losses) {
                    sum += l;
                }
                double avg_loss = sum / losses.size();
                losses.clear();
                printf("Epoch: %d, Iteration: %d, Loss: %f\n", epoch, i, avg_loss);
            }
            mlp.zero_grad();
            mlp.backward(y, y_hat);
            mlp.update(learning_rate);
        }
    }
}


void train_cuda(double learning_rate, int epoch_num, int hidden_dim, const string &dataset_path) {
    printf("Learning rate: %f, epoch number: %d, hidden dimension: %d, dataset path: %s\n", learning_rate, epoch_num, hidden_dim, dataset_path.c_str());
    // Read the MNIST dataset
    auto training_images = mnist::read_mnist_image_file<uint8_t>(dataset_path + "/train-images-idx3-ubyte");
    auto training_labels = mnist::read_mnist_label_file<uint8_t>(dataset_path + "/train-labels-idx1-ubyte");
    auto test_images = mnist::read_mnist_image_file<uint8_t>(dataset_path + "/t10k-images-idx3-ubyte");
    auto test_labels = mnist::read_mnist_label_file<uint8_t>(dataset_path + "/t10k-labels-idx1-ubyte");
    printf("Training images: %zu x %zu\n", training_images.size(), training_images[0].size());
    printf("Training labels: %zu\n", training_labels.size());
    assert(training_images.size() == training_labels.size());
    printf("Test images: %zu x %zu\n", test_images.size(), test_images[0].size());
    printf("Test labels: %zu\n", test_labels.size());
    assert(test_images.size() == test_labels.size());

    cudaError_t err;  

    // Create a neural network with 784 inputs, 100 hidden neurons and 10 outputs
    int input_dim = 784;
    int output_dim = 10;

    MLP_CUDA h_mlp_cuda;
    Init_Host_MLP(&h_mlp_cuda, input_dim, hidden_dim, output_dim);
    printf("Init host success\n");
    double *d_W1, *d_W2, *d_b1, *d_b2, *d_W1_grad, *d_W2_grad, *d_b1_grad, *d_b2_grad, *d_y1, *d_z1, *d_y2, *d_z2;
    Init_Device_MLP(&h_mlp_cuda, &d_W1, &d_W2, &d_b1, &d_b2, &d_W1_grad, &d_W2_grad, &d_b1_grad, &d_b2_grad, &d_y1, &d_z1, &d_y2, &d_z2);       //! double pointer! which means we need to pass double**, not double*
    cudaDeviceSynchronize();
    printf("Init device success\n");

    // Train the network
    for (int epoch = 0; epoch < epoch_num; epoch++) {
        vector<double> losses;
        for (int iteration = 0; iteration < training_images.size(); iteration++) {
            // printf("Iteration: %d\n", iteration);
            auto x = training_images[iteration];        // type of x: vector<unsigned char>
            auto l = training_labels[iteration];
            vector<double> y_label(output_dim, 0);
            y_label[l] = 1;
            vector<double> input = vector<double>(x.begin(),x.end());
            
            // Copy input data and labels to device memory
            double *d_input, *d_y_label;
            cudaMalloc((void **)&d_input, input.size() * sizeof(double));
            cudaMalloc((void **)&d_y_label, y_label.size() * sizeof(double));
            err = cudaMemcpy(d_input, input.data(), input.size() * sizeof(double), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {  
                printf("Failed to copy input data (x) to device: %s\n", cudaGetErrorString(err));  
            }  
            err = cudaMemcpy(d_y_label, y_label.data(), y_label.size() * sizeof(double), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {  
                printf("Failed to copy input data (y) to device: %s\n", cudaGetErrorString(err));  
            }  

            //! forward
            printf("\n-----------------------precheck-----------------------------------\n");
            printf("input dim: %i\n", input_dim);
            printf("hidden dim: %i\n", hidden_dim);
            printf("output dim: %i\n", output_dim);
            // printf("weight W1: \n");
            // for (int i = 0; i < 10; ++i) {printf("%f ", d_W1[i]);}  // only print the first 10 elements
            // //! cannot directly access d_W1 here, because it is in a device, not host
            // printf("\ngrad : \n");
            // for (int i = 0; i < 10; ++i) {printf("%f ", d_W1_grad[i]);}
            // printf("\nz2: \n");
            // for (int i = 0; i < output_dim; ++i) {printf("%f ", d_z2[i]);}

            one_layer_forward_sigmoid_kernel<<<1, hidden_dim>>>(d_input, d_W1, d_b1, d_y1, d_z1, input_dim, hidden_dim);          // input -> first hidden layer
            err = cudaGetLastError();  
            if (err != cudaSuccess) {  
                printf("Failed to launch forward kernel: %s\n", cudaGetErrorString(err));        // 从第二次循环才开始fail
                exit(0); 
            }
            printf("forward 1 success\n");

            one_layer_backward_softmax_kernel<<<1, output_dim>>>(d_y1, d_W2, d_b2, d_y2, d_z2, hidden_dim, output_dim);          // first hidden layer -> output
            softmax_normalization_kernel<<<1, 1>>>(d_z2, output_dim);          // add softmax normalization
            printf("forward 2 success\n");

            // zero grad
            set_zero_matrix_kernel<<<1, hidden_dim * input_dim>>>(d_W1, input_dim, hidden_dim);
            set_zero_matrix_kernel<<<1, hidden_dim>>>(d_b1, 1, hidden_dim);
            set_zero_matrix_kernel<<<1, output_dim * hidden_dim>>>(d_W2, hidden_dim, output_dim);
            set_zero_matrix_kernel<<<1, output_dim>>>(d_b2, 1, output_dim);
            printf("zero grad success\n");

            // backward
            one_layer_backward_softmax_kernel<<<1, output_dim>>>(d_y2, d_z2, d_y_label, d_W2_grad, d_b2_grad, hidden_dim, output_dim);          // output -> first hidden layer
            one_layer_backward_sigmoid_kernel<<<1, hidden_dim>>>(d_y1, d_W1, d_b1_grad, d_W1_grad, d_b1, d_input, input_dim, hidden_dim);          // first hidden layer -> input
            printf("backward success\n");

            // update
            matrix_update_kernel<<<1, hidden_dim * input_dim>>>(d_W1, d_W1_grad, learning_rate, input_dim, hidden_dim);
            matrix_update_kernel<<<1, hidden_dim>>>(d_b1, d_b1_grad, learning_rate, 1, hidden_dim);
            matrix_update_kernel<<<1, output_dim * hidden_dim>>>(d_W2, d_W2_grad, learning_rate, hidden_dim, output_dim);
            matrix_update_kernel<<<1, output_dim>>>(d_b2, d_b2_grad, learning_rate, 1, output_dim);
            printf("update success\n");

            // copy output data from device
            Copy_Device_to_Host(&h_mlp_cuda, d_W1, d_W2, d_b1, d_b2, d_W1_grad, d_W2_grad, d_b1_grad, d_b2_grad, d_y1, d_z1, d_y2, d_z2);

            // compute loss after device result is copied to host
            std::vector<double> y_out = std::vector<double>(h_mlp_cuda.z2, h_mlp_cuda.z2 + output_dim);
            double loss = cross_entropy(y_out, y_label);
            printf("loss: %f\n", loss);
            // Print y_label and h_mlp_cuda.z2
            // printf("y_label: ");        // 正确
            // for (int j = 0; j < y_label.size(); j++) {
            //     printf("%f ", y_label[j]);
            // }
            // printf("\n");

            // printf("h_mlp_cuda.z2: ");      // 全0，不符合预期
            // for (int j = 0; j < h_mlp_cuda.output_dim; j++) {
            //     printf("%f ", h_mlp_cuda.z2[j]);
            // }
            // printf("\n");
            losses.push_back(loss);
            // printf("we got here\n");
            if (iteration % 1000 == 0) {
                double sum = 0;
                for (auto &l: losses) {
                    sum += l;
                }
                double avg_loss = sum / losses.size();
                losses.clear();
                printf("Epoch: %d, Iteration: %d, Loss: %f\n", epoch, iteration, avg_loss);
            }
            cudaFree(d_input);
            cudaFree(d_y_label);

            // debug: observe one iteration
            exit(0);
        }
    }

    // Free device memory
    Free_Host_MLP(&h_mlp_cuda);
    Free_Device_MLP(d_W1, d_W2, d_b1, d_b2, d_W1_grad, d_W2_grad, d_b1_grad, d_b2_grad, d_y1, d_z1, d_y2, d_z2);
}

static const struct option long_options[] = {
        {"lr",      optional_argument, nullptr, 'l'},
        {"epoch",   optional_argument, nullptr, 'e'},
        {"dataset", optional_argument, nullptr, 'd'},
        {"hidden",  optional_argument, nullptr, 'h'},
        {nullptr,   no_argument,       nullptr, 0}
};

int main(int argc, char *argv[]) {
    double learning_rate = 0.001;
    int epoch_num = 10;
    string dataset_path = "../data";
    int hidden_dim = 100;
    int opt, opt_idx;
    while ((opt = getopt_long(argc, argv, "l:e:d:h:", long_options, &opt_idx)) != -1) {
        switch (opt) {
            case 'l':
                learning_rate = atof(optarg);
                break;
            case 'e':
                epoch_num = atoi(optarg);
                break;
            case 'd':
                dataset_path = optarg;
                break;
            case 'h':
                hidden_dim = atoi(optarg);
                break;
            default:
                break;
        }
    }
    // train(learning_rate, epoch_num, hidden_dim, dataset_path);
    cout << "CUDA version" << endl;
    train_cuda(learning_rate, epoch_num, hidden_dim, dataset_path);
    return 0;
}
