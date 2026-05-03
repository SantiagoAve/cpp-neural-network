#pragma once
#include <Eigen/Dense>

class ActivationFunction {
    public:
        /*
            ACTIVATION FUNCTIONS:
            These are activation functions and their derivatives. They dictate how each neuron
            gets activated. A dierivative is also necessary for doing backward in the layer.
            
            OPTIONS:
            1- "Sigmoid"
            2- "ReLU"
            3- "TanH"
        */
        static Eigen::MatrixXd sigmoid(const Eigen::MatrixXd & z);
        static Eigen::MatrixXd sigmoid_derivative(const Eigen::MatrixXd & z);

        static Eigen::MatrixXd relu(const Eigen::MatrixXd & z);
        static Eigen::MatrixXd relu_derivative(const Eigen::MatrixXd & z);

        static Eigen::MatrixXd tanh(const Eigen::MatrixXd & z); // This used a built-in function.
        static Eigen::MatrixXd tanh_derivative(const Eigen::MatrixXd & z);
};