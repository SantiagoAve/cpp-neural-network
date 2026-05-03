#pragma once
#include <Eigen/Dense>

class ActivationFunction {
    public:
        /*
            ACTIVATION FUNCTIONS:
            These are activation functions and their derivatives, meaning they store how
            each Neuron will be activated and what their derivative is like.
        */
        static Eigen::MatrixXd sigmoid(const Eigen::MatrixXd & z);
        static Eigen::MatrixXd sigmoid_derivative(const Eigen::MatrixXd & z);

        static Eigen::MatrixXd relu(const Eigen::MatrixXd & z);
        static Eigen::MatrixXd relu_derivative(const Eigen::MatrixXd & z);
};