#include <Layer.h>
#include <random>

Layer::Layer(int input_size, int output_size, const std::string & activation) {
    // Initializes weights matrix with small random values, biases vector
    // with zeros and assings the activation function.
    this->weights = Eigen::MatrixXd::Random(output_size, input_size) * 0.01;
    this->biases = Eigen::VectorXd::Zero(output_size);
    this->activation_function = activation;
}

Eigen::MatrixXd Layer::forward(const Eigen::MatrixXd & input) {
    // Calculates the basic "z = w * x + b" and applies the layer's activation
    // function.
    Eigen::MatrixXd z = (this->weights * input).colwise() + this->biases;

    if (this->activation_function == "Sigmoid") {
        return this->sigmoid(z);
    } else {
        return this->relu(z);
    }
}

Eigen::MatrixXd Layer::sigmoid(const Eigen::MatrixXd & z) {
    return 1 / ( 1 + ( -z.array() ).exp() );
}

Eigen::MatrixXd Layer::relu(const Eigen::MatrixXd & z) {
    return z.array().max(0);
}