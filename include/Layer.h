#pragma once
#include <eigen/Dense>

class Layer {
    private:
        Eigen::MatrixXd weights; // output * input in size.
        Eigen::VectorXd biases; // column/row vector.
        Eigen::MatrixXd sigmoid(const Eigen::MatrixXd & z);
    public:
        Layer(int input_size, int output_size);

        // Forward function, returns the full activation value.
        Eigen::MatrixXd forward(const Eigen::MatrixXd & input);

        // Backward function needs to be implemented too!
};