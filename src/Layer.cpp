#include "Layer.h"
#include <random>

/*
    LAYER CONSTRUCTOR:
*/
Layer::Layer(int input_size, int output_size, const std::string & activation) {
    // 0.01 avoids to have an untrainnable Neural Network.
    this->weights = Eigen::MatrixXd::Random(output_size, input_size) * 0.01;
    this->biases = Eigen::VectorXd::Zero(output_size);
    this->activation_function = activation;
}

/*
    FORWARD & BACKWARD:
*/
Eigen::MatrixXd Layer::forward(const Eigen::MatrixXd & input) {
    // These variables are going to be used for backward later on.
    this->prev_input = input;
    this->my_z = (this->weights * input).colwise() + this->biases;

    if (this->activation_function == "Sigmoid") {
        this->my_a = this->sigmoid(this->my_z);
    } else {
        this->my_a = this->relu(this->my_z);
    }

    return this->my_a;
}

Eigen::MatrixXd Layer::backward(const Eigen::MatrixXd & propag_loss_grad, double learning_rate) {
    // I know it's not the best name, but this is supposed to represent what in calculus you
    // found as 'dL/dZ = [dL/dA * dA/dZ]'.
    Eigen::MatrixXd derivative_z;

    // This is what chain rule is translated into.
    if (this->activation_function == "Sigmoid") {
        derivative_z = propag_loss_grad.array() * sigmoid_derivative(this->my_z).array();
    } else {
        derivative_z = propag_loss_grad.array() * relu_derivative(this->my_z).array();
    }

    // In this case, this is for representing 'dL/dW = [dL/dA * dA/dZ] * dZ/dW' or for biases,
    // 'dL/dB = [dL/dA * dA/dZ] * dZ/dB'. Notice how '[dL/dA * dA/dZ]' is 'derivative_z'.
    Eigen::MatrixXd derivative_w = derivative_z * this->prev_input.transpose();
    Eigen::MatrixXd derivative_b = derivative_z.rowwise().sum();

    this->weights -= learning_rate * derivative_w;
    this->biases -= learning_rate * derivative_b;

    // Keep in mind 'derivative_z' size is Output*Bach, and 'weights' size is Output*Input,
    // therefore, transposition is necessary for 'weights'.
    return this->weights.transpose() * derivative_z;
}

/*
    SIGMOID FUNCTION:
*/
Eigen::MatrixXd Layer::sigmoid(const Eigen::MatrixXd & z) {
    return 1 / ( 1 + ( -z.array() ).exp() );
}

Eigen::MatrixXd Layer::sigmoid_derivative(const Eigen::MatrixXd & z) {
    Eigen::MatrixXd sigmoid_result = this->sigmoid(z);
    return sigmoid_result.array() * (1 - sigmoid_result.array());
}

/*
    RELU FUNCTION:
*/
Eigen::MatrixXd Layer::relu(const Eigen::MatrixXd & z) {
    return z.array().max(0);
}

Eigen::MatrixXd Layer::relu_derivative(const Eigen::MatrixXd & z) {
    return (z.array() > 0).cast<double>();
}