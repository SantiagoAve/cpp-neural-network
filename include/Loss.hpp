#pragma once
#include <Eigen/Dense>

namespace nn {
    /*
        LOSS FUNCTIONS:
        These functions measure how wrong the network predictions are; a bigger number means a
        worse prediction. Its gradient with respect to the prediction is the seed for backprop,
        as it tells you the direction of maximum growth. We want the opposite way, of course.

        OPTIONS:
        1. "MSE" (Mean Squared Error).
        2. "Cross-Entropy" (Planned, together with a softmax output layer).
    */
    class Loss {
        public:
            /*
                MSE:
                This is the MSE averaged over all matrix elements, so it's:
                    L(Y, P) = mean((Y - P) ^ 2)
                Also, throws and exception, std::invalid_argument, when shapes differ.
            */
            [[nodiscard]] static double mean_squared(const Eigen::MatrixXd & true_values,
                                                     const Eigen::MatrixXd & predicted_values);
            
            /*
                MSE GRADIENT:
                This is the gradient of the MSE with respect to the predictions, so it's:
                    dL/dP = 2 * (P - Y) / element_count
                You divide by the element count, not just the batch size, so this works fine
                for any output dimensionality.
            */
            [[nodiscard]] static Eigen::MatrixXd mean_squared_derivative(const Eigen::MatrixXd & true_values,
                                                                         const Eigen::MatrixXd & predicted_values);
    };
} // namespace nn
