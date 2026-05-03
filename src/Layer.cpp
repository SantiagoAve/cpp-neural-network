#include "Layer.hpp"
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
    -----------------------------------------------------------------------------------------------
*/

/*
    FORWARD & BACKWARD:
*/
Eigen::MatrixXd Layer::forward(const Eigen::MatrixXd & input) {
    // These variables are going to be used for backward later on.
    this->prev_input = input;
    this->layer_z = (this->weights * input).colwise() + this->biases;

    if (this->activation_function == "ReLU") {
        this->layer_a = ActivationFunction::relu(this->layer_z);
    } else {
        this->layer_a = ActivationFunction::sigmoid(this->layer_z);
    }

    return this->layer_a;
}

Eigen::MatrixXd Layer::backward(const Eigen::MatrixXd & propag_loss_grad, double learning_rate) {
    // I know it's not the best name, but this is supposed to represent what in calculus you
    // found as 'dL/dZ = [dL/dA * dA/dZ]'.
    Eigen::MatrixXd derivative_z;

    // This is what chain rule is translated into.
    if (this->activation_function == "ReLU") {
        derivative_z = propag_loss_grad.array()
                     * ActivationFunction::relu_derivative(this->layer_z).array();
    } else {
        derivative_z = propag_loss_grad.array()
                     * ActivationFunction::sigmoid_derivative(this->layer_z).array();
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
    -----------------------------------------------------------------------------------------------
*/