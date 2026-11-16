#include "Network.hpp"

#include "Loss.hpp"
#include <format>
#include <stdexcept>

namespace nn {
    /*
        NETWORK CONSTRUCTOR:
        Takes ownership by moving, so the caller's vectors end up empty and therefore makes the old
        double-ownership bug impossible!
    */
    Network::Network(std::vector<std::unique_ptr<Layer>> layers, std::unique_ptr<Optimizer> optimizer)
        : layers_(std::move(layers)), optimizer_(std::move(optimizer)) {
        if (!optimizer_) {
            throw std::invalid_argument("Network: optimizer must not be null");
        }
        for (const auto & layer : layers_) {
            if (!layer) {
                throw std::invalid_argument("Network: layers must not be null");
            }
        }
    }
    /*
        -----------------------------------------------------------------------------------------------
    */

    // Builder design pattern.
    void Network::add(std::unique_ptr<Layer> layer) {
        if (!layer) {
            throw std::invalid_argument("Network::add: layer must not be null");
        }
        layers_.push_back(std::move(layer));
    }

    /*
        FORWARD:
        This function chains every layer in order, applying the forward function to them.
    */
    Eigen::MatrixXd Network::forward(const Eigen::MatrixXd & input) {
        Eigen::MatrixXd output = input;
        for (auto & layer : layers_) {
            output = layer->forward(output);
        }
        return output;
    }
    /*
        -----------------------------------------------------------------------------------------------
    */

    /*
        BACKWARD:
        This function backpropagates dL/d(output) from the last layer to the first in reverse
        order. Then, lets the optimizer apply every layer's stored gradients.
    */
    void Network::backward(const Eigen::MatrixXd & true_values, const Eigen::MatrixXd & predicted_values) {
        Eigen::MatrixXd gradient = Loss::mean_squared_derivative(true_values, predicted_values);

        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
            gradient = (*it)->backward(gradient);
        }

        for (auto & layer : layers_) {
            optimizer_->update(*layer);
        }
    }
    /*
        -----------------------------------------------------------------------------------------------
    */

    /*
        LOSS FUNCTION:
        This function applies the loss function between the true_values and the predicted ones by the net.
    */
    double Network::loss(const Eigen::MatrixXd & true_values, const Eigen::MatrixXd & predicted_values) {
        return Loss::mean_squared(true_values, predicted_values);
    }
    /*
        -----------------------------------------------------------------------------------------------
    */
} // namespace nn
