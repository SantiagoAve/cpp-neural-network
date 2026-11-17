#pragma once
#include "Layer.hpp"
#include "Optimizer.hpp"
#include <cstddef>
#include <memory>
#include <vector>

namespace nn {
    /*
        NETWORK:
        A network is a ordered sequence of Layers and one single Optimizer. An average training
        step is: forward -> loss -> backward; the most interesting one being backward, which
        backpropagates the loss gradient thorugh every layer, letting the optimizer update
        each one later on.

        Here Layers are owned through a unique_ptr, enabling polymorphism, so any present or
        future Layer subtype just work perfectly, and guaranting stable adresses that Adam
        relies on.
    */
    class Network {
        public:
            /*
                NETWORK CONSTRUCTOR:
                Takes ownership of the layers and the optimizer. Before, it used to create a
                copy of the arguments, now it works fine while passing by value.
            */
            Network(std::vector<std::unique_ptr<Layer>> layers, std::unique_ptr<Optimizer> optimizer);
            
            /*
                ADD:
                This is a convenience builder; simply appends one more layer to the end of the
                stack. That way, you can keep adding Layers with ease.
            */
            void add(std::unique_ptr<Layer> layer);

            /*
                FORWARD:
                This simply runs the forward pass on every layer of the network, in order. 
            */
            Eigen::MatrixXd forward(const Eigen::MatrixXd & input);

            /*
                BACKWARD:
                This backpropagates the error, dL/d(output), computed by the error formula between
                true_values and predicted_values, and then applies the optimizer to every layer.
                Note that predicted values must come from the most recent call to forward().
            */
            void backward(const Eigen::MatrixXd & true_values, const Eigen::MatrixXd & predicted_values);

            /*
                LOSS FORMULA:
                This computes the current loss function of the network based on a (true, predicted) pair.
            */
            [[nodiscard]] static double loss(const Eigen::MatrixXd & true_values,
                                             const Eigen::MatrixXd & predicted_values);

            // Simple size getter.
            [[nodiscard]] std::size_t size() const { return layers_.size(); }

            /*
                TEST FUNCTIONS:
                These are functions used in testing, logging or inspection.
            */
            // Read-only access.
            [[nodiscard]] const Layer & layer(std::size_t index) const { return *layers_.at(index); }

            // Mutable access (gradient checks).
            [[nodiscard]] Layer & layer(std::size_t index) { return *layers_.at(index); }

        private:
            /*
                INNER VARIABLES:
                These are mainly pointers to the network components.
            */
            std::vector<std::unique_ptr<Layer>> layers_;
            std::unique_ptr<Optimizer> optimizer_;
    };
} // namespace nn
