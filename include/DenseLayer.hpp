#pragma once
#include "Layer.hpp"
#include "Activation.hpp"
#include <cstdint>
#include <optional>

namespace nn {
    /*
        DENSE (OR FULLY CONNECTED) LAYER:
        The classic layer for NN. This implements A = sigma(W * Z + b), where W is of size
        output_size x input_size; and b is just of output_size. Also, inputs and outputs are
        batches stored column-wise; which is features x batch_size.

        Note that weight initialization is matched to the activation type; He for ReLU family
        and Xavier for Tanh or Sigmoid. This keeps activation and gradient variance stable
        across all layers. And a fixed seed makes construction deterministic, granting the same
        results no matter when it's run.
    */
    class DenseLayer final : public Layer {
        public:
            /*
                DENSELAYER CONSTRUCTOR:
                Creates a DenseLayer. Note that input_size and output_size must be both positive.
                There's also the possibility to add a seed, which fixes the random initialization,
                making it deterministic; useful for reproducible runs or tests.
            */
            DenseLayer(int input_size, int output_size, const Activation & activation,
                    std::optional<std::uint32_t> seed = std::nullopt);

            Eigen::MatrixXd forward(const Eigen::MatrixXd & input) override;
            Eigen::MatrixXd backward(const Eigen::MatrixXd & output_gradient) override;

            [[nodiscard]] std::string name() const override;
            Parameters parameters() override;

        private:
            /*
                INNER VARIABLES:
                These variables are used to store information of the layer.
            */
            int input_size_;
            int output_size_;

            Activation activation_;

            Eigen::MatrixXd weights_; // Size: output_size x input_size.
            Eigen::VectorXd biases_; // Size: output_size.
            Eigen::MatrixXd grad_weights_; // Size: output_size x input_size.
            Eigen::VectorXd grad_biases_; // Size: output_size.

            Eigen::MatrixXd pre_activation_; // This is z = W * input + b, cached for backward pass.
            Eigen::MatrixXd last_input_; // Also cached for backward pass.
    };
} // namespace nn
