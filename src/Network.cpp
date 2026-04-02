#include "Network.h"

Network::Network(const std::vector<Layer> & layers) : layers(layers) {}

Eigen::MatrixXd Network::net_forward(const Eigen::MatrixXd & input) {
    Eigen::MatrixXd cur_output = input;
    for (auto & layer : layers) {
        cur_output = layer.forward(cur_output);
    }
    return cur_output;
}