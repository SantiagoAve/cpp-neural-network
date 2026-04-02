#pragma once
#include <eigen/Dense>
#include <string>

class Layer {
    public:
        Layer(int input_size, int output_size, const std::string & activation = "Sigmoid");

        // Forward function that returns a Matrix of dimensions Output*Bach.
        // Each component is corresponding to the activation of a Neuron.
        Eigen::MatrixXd forward(const Eigen::MatrixXd & input);

        // Backward function needs to be implemented too!
    private:
        Eigen::MatrixXd weights; // (output * input in size)
        Eigen::VectorXd biases; // (column/row vector)
        std::string activation_function; // (Sigmoid or ReLU)
        // Activation functions:
        Eigen::MatrixXd sigmoid(const Eigen::MatrixXd & z);
        Eigen::MatrixXd relu(const Eigen::MatrixXd & z);
};