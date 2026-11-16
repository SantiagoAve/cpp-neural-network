#include "Activation.hpp"

/*
    IMPORTANT IMPLEMENTATION NOTE:
    Every lambda below declares an explicit '-> Eigen::MatrixXd' return type. With a deduced return
    type, the lambda would return an Eigen EXPRESSION TEMPLATE referencing its local temporaries;
    those temporaries die when the lambda returns, leaving the expression dangling, and the conversion
    to MatrixXd then happens at the std::function boundary on freed memory. That is undefined behavior
    that typically "works" at -O0 and corrupts data at -O3. The explicit return type forces the
    expression to be materialized inside the lambda, while everything is still alive.
*/

namespace nn {
    /*
        SIGMOID (LOGISTIC):
        This is sigma(z) = 1 / (1 + e^{-z}). Numerically well behaved in double precision:
        for very negative z the exponential overflows to infinity, but 1 / (1 + inf) still
        evaluates to the correct limit 0.
    */
    Activation Activation::sigmoid() {
        Activation a;
        a.function = [](const Eigen::MatrixXd & z) -> Eigen::MatrixXd {
            return (1.0 / (1.0 + (-z.array()).exp())).matrix();
        };
        a.derivative = [](const Eigen::MatrixXd& z) -> Eigen::MatrixXd {
            const Eigen::ArrayXXd s = 1.0 / (1.0 + (-z.array()).exp());
            return (s * (1.0 - s)).matrix();
        };
        a.init = Init::Xavier;
        return a;
    }
    /*
        -----------------------------------------------------------------------------------------------
    */

    /*
        ReLU (RECTIFIED LINEAR UNIT):
        This is sigma(z) = max(z, 0). Its derivative is 1 for z > 0 and 0 for z < 0; at
        exactly z = 0 the sub-gradient convention is 0 (which is the same as PyTorch).
    */
    Activation Activation::relu() {
        Activation a;
        a.function = [](const Eigen::MatrixXd & z) -> Eigen::MatrixXd {
            return z.array().max(0.0).matrix();
        };
        a.derivative = [](const Eigen::MatrixXd & z) -> Eigen::MatrixXd {
            return (z.array() > 0.0).cast<double>().matrix();
        };
        a.init = Init::He;
        return a;
    }
    /*
        -----------------------------------------------------------------------------------------------
    */

    /*
        HYPERBOLIC TANGENT:
        This is sigma(z) = tanh(z). Thanks to that, its derivative can be easily implemented.
        Plus, Eigen ships a vectorized tanh that works directly on arrays!
    */
    Activation Activation::tanh() {
        Activation a;
        a.function = [](const Eigen::MatrixXd & z) -> Eigen::MatrixXd {
            return z.array().tanh().matrix();
        };
        a.derivative = [](const Eigen::MatrixXd & z) -> Eigen::MatrixXd {
            const Eigen::ArrayXXd t = z.array().tanh();
            return (1.0 - t.square()).matrix();
        };
        a.init = Init::Xavier;
        return a;
    }
    /*
        -----------------------------------------------------------------------------------------------
    */

    /*
        IDENTITY (LINEAR):
        This is sigma(z) = z. Useful for regression output layers and for unit-testing the
        optimizer logic on problems with a closed-form solution.
    */
    Activation Activation::identity() {
        Activation a;
        a.function = [](const Eigen::MatrixXd& z) -> Eigen::MatrixXd {
            return z;
        };
        a.derivative = [](const Eigen::MatrixXd& z) -> Eigen::MatrixXd {
            return Eigen::MatrixXd::Ones(z.rows(), z.cols());
        };
        a.init = Init::Xavier;
        return a;
    }
    /*
        -----------------------------------------------------------------------------------------------
    */
} // namespace nn
