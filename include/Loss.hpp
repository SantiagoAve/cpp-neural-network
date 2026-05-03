#pragma once
#include <Eigen/Dense>

class Loss {
    public:
        /*
            FORMULAS:
            1. Mean Squared Error formula.
            2. Mean Squared Error derivative.
            3. Cross-Entropy Loss formula. (COMING SOON)
            4. Cross-Entropy Loss derivative. (COMING SOON)
        */
        static double mean_squared(const Eigen::MatrixXd & true_values,
                                   const Eigen::MatrixXd & predicted_values);
                                   
        static Eigen::MatrixXd mean_derivative(const Eigen::MatrixXd & true_values,
                                               const Eigen::MatrixXd & predicted_values);
};