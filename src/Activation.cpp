#include "ActivationF.hpp"

/*
    SIGMOID FUNCTION:
*/
Eigen::MatrixXd ActivationFunction::sigmoid(const Eigen::MatrixXd & z) {
    return 1 / ( 1 + ( -z.array() ).exp() );
}

Eigen::MatrixXd ActivationFunction::sigmoid_derivative(const Eigen::MatrixXd & z) {
    Eigen::MatrixXd sigmoid_result = sigmoid(z);
    return sigmoid_result.array() * (1 - sigmoid_result.array());
}
/*
    -----------------------------------------------------------------------------------------------
*/

/*
    RELU FUNCTION:
*/
Eigen::MatrixXd ActivationFunction::relu(const Eigen::MatrixXd & z) {
    return z.array().max(0);
}

Eigen::MatrixXd ActivationFunction::relu_derivative(const Eigen::MatrixXd & z) {
    // If an element is > 0, then it puts a true, else a false.
    // Then, casted to a double for better efficiency in Eigen.
    return (z.array() > 0).cast<double>();
}