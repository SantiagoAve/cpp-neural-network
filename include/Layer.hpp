#pragma once
#include <Eigen/Dense>
#include <string>

namespace nn {
    /*
        PARAMETERS:
        This is a lightweight and non-owning view of a layer's trainable tensors. Optimizers
        receive this view; there's no touching layer internals with getters. If a layer is
        parameterless, this returns an empty view, which optimizers simply skip.
    */
    struct Parameters {
        Eigen::MatrixXd * weights = nullptr;
        Eigen::VectorXd * biases = nullptr;
        Eigen::MatrixXd * grad_weights = nullptr;
        Eigen::VectorXd * grad_biases = nullptr;

        [[nodiscard]] bool empty() const { return weights == nullptr; }
    };

    /*
        LAYER (ABSTRACT BASE):
        A Layer can be understood as one step on a network's transformation. That means that
        concrete layers must implement a forward pass and its backward, or gradient, pass.

        This interface is left deliberately general on purpose, DenseLayer, Conv2D, and other
        layers can be added later without touching anything else. This is basic polymorphism
        and herency.

        Note that both passes are non-const, this is because they cache internal state; that
        also means that forward passes must always come before backward ones, since they use
        said cache.
    */
    class Layer {
        public:
            // A virtual destructor, needed for polymorphism.
            virtual ~Layer() = default;

            /*
                FORWARD ACTION:
                Computes the layer output for a batch of column vectors; size is features x batch_size.
                It also throws an exception, std::invalid_argument, when there's a shape mismatch; that
                way it doesn't cause any undefined undefined behavior.
            */
            virtual Eigen::MatrixXd forward(const Eigen::MatrixXd & input) = 0;

            /*
                BACKWARD ACTION:
                Given the gradient with respect to this layer's output, dL/dA, computes and stores dL/dW
                and dL/dB for the optimizer, and returns dL/dInput, so the previous layer can continue
                the chain rule. That's Calculus 3 right there.
            */
            virtual Eigen::MatrixXd backward(const Eigen::MatrixXd & output_gradient) = 0;

            /*
                PARAMETER RETURN:
                Returns a view on the parameters, or nothing for parameterless layers.
            */
            virtual Parameters parameters() { return {}; }

            // This is a readable description, used for logging and debugging purposes.
            [[nodiscard]] virtual std::string name() const = 0;
    };
} // namespace nn
