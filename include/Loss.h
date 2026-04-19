#pragma once
#include <Eigen/Dense>

class Loss {
    public:
        /*
            FORMULAS:
            1. Mean Squared Error formula.
            2. Mean Squared Error derivative.
        */
        static double mean_squared(const Eigen::MatrixXd & true_values,
                                   const Eigen::MatrixXd & predicted_values);

        static Eigen::MatrixXd mean_derivative(const Eigen::MatrixXd & true_values,
                                               const Eigen::MatrixXd & predicted_values);
};