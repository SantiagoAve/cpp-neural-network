#pragma once
#include <Eigen/Dense>

class Loss {
    public:
        // Mean Squared Error formula.
        static double mean_squared(const Eigen::MatrixXd & true_values, const Eigen::MatrixXd & predicted_values);

        // Mean Squared Error derivative.
        static Eigen::MatrixXd mean_derivative(const Eigen::MatrixXd & true_values, const Eigen::MatrixXd & predicted_values);
};