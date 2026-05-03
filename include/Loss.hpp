#pragma once
#include <Eigen/Dense>

class Loss {
    public:
        /*
            LOSS FUNCTIONS:
            These are functions that helps to visualize how wrong predictions made by the NN are.
            The bigger the number, the worse the prediction. It's derivative (Or gradient in this
            case) tells you which direction gives you more error, so its opossite direction is
            the desired direction.

            OPTIONS:
            1. "MSE" (Mean Squared Error)
            2. "Cross-Entropy" (COMING SOON)
        */
        static double mean_squared(const Eigen::MatrixXd & true_values,
                                   const Eigen::MatrixXd & predicted_values);
                                   
        static Eigen::MatrixXd mean_derivative(const Eigen::MatrixXd & true_values,
                                               const Eigen::MatrixXd & predicted_values);
};