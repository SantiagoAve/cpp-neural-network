#pragma once

#include <Eigen/Dense>
#include <functional>
#include <string>

namespace nn {
    /*
        ACTIVATION:
        This structure contains an activation function, along its derivative and best
        initialization scheme that matches it based on statistics. He for ReLU family and
        Xavier for Tanh or Sigmoid. This is better than the older class because it doesn't
        let you pair a function with a wrong derivative.

        Kind serves as a lightweight tag, showing which activation is being used. Note that
        a std::function cannot be serialized, and this tag keeps the information for further
        use in the future. Plus, it also doubles as a readable name for logging and debugging.
    */
    struct Activation {
        using Function = std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)>;

        enum class Init { He, Xavier };
        enum class Kind { Relu, Sigmoid, Tanh, Identity };

        Function function; // This would be textbook sigma(z).
        Function derivative; // And this would be sigma'(z); evaluated on z.
        Init init = Init::Xavier; // Matching weight initialization scheme.
        Kind kind = Kind::Identity; // Serializable identity of this activation.

        // Named factories:
        static Activation relu();
        static Activation sigmoid();
        static Activation tanh();
        static Activation identity(); // This is regular linear (Useful for regression outputs).

        /*
            FROM KIND:
            Rebuilds a named activation from its tag; used when loading a model from disk.
            Unknown tags throw std::invalid_argument.
        */
        [[nodiscard]] static Activation from_kind(Kind kind);

        /*
            LABEL / PARSE:
            Converts between the tag and its text form ("relu", "sigmoid", "tanh", "identity"),
            which is the form stored inside model files.
        */
        [[nodiscard]] static const char * label(Kind kind);
        [[nodiscard]] static Kind parse(const std::string & name);
    }; // struct Activation
} // namespace nn
