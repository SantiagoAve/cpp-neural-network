#include "Loss.h"

/*
    FORMULAS:
*/
double Loss::mean_squared(const Eigen::MatrixXd & true_values,
                          const Eigen::MatrixXd & predicted_values) {
    Eigen::MatrixXd value_difference = true_values - predicted_values;
    return value_difference.array().square().mean();
}

// Mean Squared Error derivative.
Eigen::MatrixXd Loss::mean_derivative(const Eigen::MatrixXd & true_values,
                                      const Eigen::MatrixXd & predicted_values) {
    return -2 * (true_values - predicted_values) / true_values.rows();
}
/*
    -----------------------------------------------------------------------------------------------
*/