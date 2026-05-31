#pragma once
#include <vector>
#include "Layer.hpp"
#include "Loss.hpp"
#include "Optimizer.hpp"

class Network {
    public:
        /*
            NETWORK CONSTRUCTOR:
            Initializes a Network. Currently, Layer creation is tied to user,
            future updates will allow Layer creation to be done here.
        */
        Network(const std::vector<Layer> & layers, std::unique_ptr<Optimizer> ptr_to_optimizer)
                : layers(layers), num_layers(layers.size()), optimizer(std::move(ptr_to_optimizer)) {};

        /*
            FORWARD ACTION:
            Funcion that executes forward on each layer, providing a final output,
            starts in the first layer, ends in the last one.
        */
        Eigen::MatrixXd net_forward(const Eigen::MatrixXd & input);

        /*
            BACKWARD ACTION:
            Function that executes backward on each layer, correcting biases and
            weights, beginning from the end and moving backwards.
        */
       void net_backward(const Eigen::MatrixXd & true_values, const Eigen::MatrixXd & predicted_values);

    private:
        /*
            INNER VARIABLES:
            These variables store useful information about the network.
        */
        std::vector<Layer> layers;
        size_t num_layers;
        std::unique_ptr<Optimizer> optimizer;
};