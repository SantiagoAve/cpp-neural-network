#pragma once
#include <vector>
#include "Layer.h"

class Network {
    public:
        // Constructor or class. Initializes a Network with a list (Vector)
        // of prebuilt Layers.
        Network(const std::vector<Layer> & layers);

        // Forward function that executes forward in each layer.
        Eigen::MatrixXd net_forward(const Eigen::MatrixXd & input);
    private:
        std::vector<Layer> layers;
};