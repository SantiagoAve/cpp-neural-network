#pragma once
#include <Eigen/Dense>
#include <functional>

namespace nn {
    /*
        ACTIVATION:
        This structure contains an activation function, along its derivative and
        best initialization scheme that matches it based on statistics. He for ReLU
        family and Xavier for Tanh or Sigmoid. This is better than the older class
        because it doesn't let you pair a function with a wrong derivative.
    */
    struct Activation {
        using Function = std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)>;

        enum class Init { He, Xavier };

        Function function; // This would be textbook sigma(z).
        Function derivative; // And this would be sigma'(z); evaluated on z.
        Init init = Init::Xavier; // Matching weight initialization scheme.

        // Named factories:
        static Activation relu();
        static Activation sigmoid();
        static Activation tanh();
        static Activation identity(); // This is regular linear (useful for regression outputs).
    };
} // namespace nn
