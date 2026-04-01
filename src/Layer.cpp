#include <Layer.h>
#include <random>

Layer::Layer(int input_size, int output_size) {
    // Initializes weights matrix with small random values.
    // Initializes biases vector with zeros.
    this->weights = Eigen::MatrixXd::Random(output_size, input_size) * 0.01;
    this->biases = Eigen::VectorXd::Zero(output_size);
}

Eigen::MatrixXd Layer::forward(const Eigen::MatrixXd & input) {
    // Calculates the basic "z = w * x + b" and applies Sigmoid to it.
    Eigen::MatrixXd z = (this->weights * input).colwise() + this->biases;
    return this->sigmoid(z);
}

Eigen::MatrixXd Layer::sigmoid(const Eigen::MatrixXd & z) {
    return 1 / ( 1 + ( -z.array() ).exp() );
}