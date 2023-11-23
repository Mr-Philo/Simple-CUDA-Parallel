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
    Host_MLP_CUDA h_mlp_cuda(784, hidden_dim, 10);
    Device_MLP_CUDA d_mlp_cuda(784, hidden_dim, 10);
    copyDataToDevice(h_mlp_cuda, d_mlp_cuda);

    // Train the network
    for (int epoch = 0; epoch < epoch_num; epoch++) {
        vector<double> losses;
        for (int i = 0; i < training_images.size(); i++) {
            printf("Iteration: %d\n", i);
            auto x = training_images[i];        // type of x: vector<unsigned char>
            auto l = training_labels[i];
            vector<double> y_label(10, 0);
            y_label[l] = 1;
            vector<double> input = vector<double>(x.begin(),x.end());

            // Copy input data and labels to device memory
            double *d_x, *d_y;
            cudaMalloc((void **)&d_x, input.size() * sizeof(double));
            cudaMalloc((void **)&d_y, y_label.size() * sizeof(double));
            cudaMemcpy(d_x, input.data(), input.size() * sizeof(double), cudaMemcpyHostToDevice);
            err = cudaMemcpy(d_y, y_label.data(), y_label.size() * sizeof(double), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {  
                printf("Failed to copy input data to device: %s\n", cudaGetErrorString(err));  
            }  

            // run the main CUDA kernel
            int num_of_threads = 256;
            int input_size = input.size();
            dim3 block_size(num_of_threads);
            dim3 num_of_blocks((input_size + num_of_threads - 1) / num_of_threads);
            train_mlp_cuda<<<num_of_blocks, block_size>>>(d_mlp_cuda, d_x, d_y, learning_rate);
            err = cudaGetLastError();  
            if (err != cudaSuccess) {  
                printf("Failed to launch train_mlp_cuda kernel: %s\n", cudaGetErrorString(err));        // 从第二次循环才开始fail
                exit(0); 
            }
            // compute loss
            cudaMemcpy(h_mlp_cuda.z2.data(), d_mlp_cuda.d_z2, h_mlp_cuda.z2.size() * sizeof(double), cudaMemcpyDeviceToHost);
            
            auto loss = cross_entropy(y_label, h_mlp_cuda.z2);
            // Print y_label and h_mlp_cuda.z2
            printf("y_label: ");        // 正确
            for (int j = 0; j < y_label.size(); j++) {
                printf("%f ", y_label[j]);
            }
            printf("\n");

            printf("h_mlp_cuda.z2: ");      // 全0，不符合预期
            for (int j = 0; j < h_mlp_cuda.z2.size(); j++) {
                printf("%f ", h_mlp_cuda.z2[j]);
            }
            printf("\n");
            losses.push_back(loss);
            // printf("we got here\n");
            if (i % 1000 == 0) {
                double sum = 0;
                for (auto &l: losses) {
                    sum += l;
                }
                double avg_loss = sum / losses.size();
                losses.clear();
                printf("Epoch: %d, Iteration: %d, Loss: %f\n", epoch, i, avg_loss);
            }
            cudaFree(d_x);
            cudaFree(d_y);
        }
    }

    // Free device memory
    freeDeviceMemory(d_mlp_cuda);
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
