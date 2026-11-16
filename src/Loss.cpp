#include "Loss.hpp"
#include <format>
#include <stdexcept>

namespace nn {
    /*
        MSE (Mean Squared Error):
        Averaged over every element of the batch, so the value is independent of batch size
        and output dimensionality.
    */
    double Loss::mean_squared(const Eigen::MatrixXd & true_values,
                            const Eigen::MatrixXd & predicted_values) {
        if (true_values.rows() != predicted_values.rows() ||
            true_values.cols() != predicted_values.cols()) {
            throw std::invalid_argument(std::format(
                "Loss::mean_squared: shape mismatch ({}x{} vs {}x{})",
                true_values.rows(), true_values.cols(),
                predicted_values.rows(), predicted_values.cols()));
        }

        const Eigen::MatrixXd difference = true_values - predicted_values;
        return difference.array().square().mean();
    }

    /*
        MSE GRADIENT (With respect to predictions):
        With L = (1 / N) * sum((Y - P) ^ 2), over all N = rows * cols elements;

            dL/d(P_ij) = -2 * (Y_(ij) - P_(ij)) / N -> 2 * (P_(ij) - Y_(ij)) / N.

        The division by N (Number of elements) is what keeps this consistent with mean_squared
        for multi-dimensional outputs.
    */
    Eigen::MatrixXd Loss::mean_squared_derivative(const Eigen::MatrixXd & true_values,
                                                const Eigen::MatrixXd & predicted_values) {
        if (true_values.rows() != predicted_values.rows() ||
            true_values.cols() != predicted_values.cols()) {
            throw std::invalid_argument(std::format(
                "Loss::mean_squared_derivative: shape mismatch ({}x{} vs {}x{})",
                true_values.rows(), true_values.cols(),
                predicted_values.rows(), predicted_values.cols()));
        }

        const auto element_count = static_cast<double>(true_values.size());
        return 2.0 * (predicted_values - true_values) / element_count;
    }
    /*
        -----------------------------------------------------------------------------------------------
    */
} // namespace nn
