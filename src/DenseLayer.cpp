#include "DenseLayer.hpp"

#include <cmath>
#include <format>
#include <random>
#include <stdexcept>

namespace nn {
    namespace {
        /*
            RNG FUNCTION:
            This function works deterministic when a seed is given (For tests or demos), in any other
            case, its random.
        */
        std::mt19937 make_rng(std::optional<std::uint32_t> seed) {
            if (seed.has_value()) {
                return std::mt19937(*seed);
            }
            return std::mt19937(std::random_device{}());
        }
        /*
            -----------------------------------------------------------------------------------------------
        */
    } // (Anonymous namespace)

    /*
        DENSE LAYER CONSTRUCTOR:
        Note that weights are drawn from a Gaussian distribution, whose spread matches the activarion:

            He: N(0, sqrt(2 / fan_in)) -> ReLU family.
            Xavier: N(0, sqrt(2 / (fan_in + fan_out))) -> tanh / sigmoid.

        Keeping the variance of activations and gradients somewhtat constant across layers is what makes
        them trainable at all. Otherwise, a small variance would shrink the signal layer by layer and one
        too hight would saturate the network.
    */
    DenseLayer::DenseLayer(int input_size, int output_size, const Activation & activation,
                        std::optional<std::uint32_t> seed)
        : input_size_(input_size), output_size_(output_size), activation_(activation) {
        if (input_size_ <= 0 || output_size_ <= 0) {
            throw std::invalid_argument(
                std::format("DenseLayer: sizes must be positive (got {} -> {})",
                            input_size_, output_size_));
        }

        const double std_dev = activation_.init == Activation::Init::He
            ? std::sqrt(2.0 / static_cast<double>(input_size_))
            : std::sqrt(2.0 / static_cast<double>(input_size_ + output_size_));

        std::mt19937 rng = make_rng(seed);
        std::normal_distribution<double> normal(0.0, std_dev);

        weights_ = Eigen::MatrixXd::NullaryExpr(output_size_, input_size_,
            [&normal, &rng](Eigen::Index, Eigen::Index) {
                return normal(rng);
            });

        biases_ = Eigen::VectorXd::Zero(output_size_);
        grad_weights_ = Eigen::MatrixXd::Zero(output_size_, input_size_);
        grad_biases_ = Eigen::VectorXd::Zero(output_size_);
    }
    /*
        -----------------------------------------------------------------------------------------------
    */

    /*
        These two functions are for debugging and logging purposes.
    */
    std::string DenseLayer::name() const {
        return std::format("Dense({} -> {})", input_size_, output_size_);
    }

    Parameters DenseLayer::parameters() {
        return Parameters{&weights_, &biases_, &grad_weights_, &grad_biases_};
    }
    /*
        -----------------------------------------------------------------------------------------------
    */

    /*
        FORWARD:
        This is z = W * X + B (One column per batch element) and then A = sigma(z). Both z and X are
        cached, so its used later on during backward and avoids recomputation. Small memory price.
    */
    Eigen::MatrixXd DenseLayer::forward(const Eigen::MatrixXd & input) {
        if (input.rows() != input_size_) {
            throw std::invalid_argument(std::format(
                "DenseLayer::forward: input has {} row(s), expected {} "
                "(rows are features, columns are batch elements)",
                input.rows(), input_size_));
        }

        last_input_ = input;
        pre_activation_ = (weights_ * input).colwise() + biases_;
        return activation_.function(pre_activation_);
    }
    /*
        -----------------------------------------------------------------------------------------------
    */

    /*
        BACKWARD:
        Given dL/dA, the chain rule gives, element-wise:

            dL/dZ = dL/dA . sigma'(Z)
            dL/dW = dL/dZ * X ^ T (sums the batch dimension)
            dL/dB = row - sum(dL/dZ) (sums the batch dimension)
            dL/dX = W ^ T * dL/dZ (returned, feeds the previous layer)

        Only the gradients are stored here; updating the parameters is the Optimizer's job.
    */
    Eigen::MatrixXd DenseLayer::backward(const Eigen::MatrixXd & output_gradient) {
        if (output_gradient.rows() != output_size_) {
            throw std::invalid_argument(std::format(
                "DenseLayer::backward: gradient has {} row(s), expected {}",
                output_gradient.rows(), output_size_));
        }

        const Eigen::MatrixXd grad_z = output_gradient.array()
                                    * activation_.derivative(pre_activation_).array();

        grad_weights_ = grad_z * last_input_.transpose();
        grad_biases_ = grad_z.rowwise().sum();

        return weights_.transpose() * grad_z;
    }
    /*
        -----------------------------------------------------------------------------------------------
    */
} // namespace nn
