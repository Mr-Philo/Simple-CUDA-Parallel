#include <unistd.h>
#include <iostream>
#include <string>
#include <getopt.h>
#include <cassert>
#include "utils.h"
#include "mnist_reader_less.h"
#include "mlp.h"
#include "functions.h"

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
    train(learning_rate, epoch_num, hidden_dim, dataset_path);
    return 0;
}
