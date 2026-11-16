#include "Activation.hpp"
#include "DenseLayer.hpp"
#include "Network.hpp"
#include "Optimizer.hpp"

#include <Eigen/Dense>
#include <format>
#include <iostream>
#include <memory>
#include <print>
#include <vector>

/*
    XOR DEMO:
    The "hello world" of neural networks! Learn the XOR truth table with a network layout such
    as 2 -> 4 -> 1 (ReLU hidden layer, sigmoid output), trained with Adam on two full batches.
    Fixed seeds make the run reproducible: every machine, every compiler, the same numbers.
*/
int main() {
    /*
        Again, network: 2 -> 4 -> 1. Adam with its paper-recommended defaults.

        The fixed seeds are not decoration: with ReLU hidden units, an unlucky initialization can
        leave every hidden neuron dead and the network stuck at 0.5 forever (Known as "dead ReLU" failure).
    */ 
    std::vector<std::unique_ptr<nn::Layer>> layers;
    layers.push_back(std::make_unique<nn::DenseLayer>(2, 4, nn::Activation::relu(), 4u));
    layers.push_back(std::make_unique<nn::DenseLayer>(4, 1, nn::Activation::sigmoid(), 104u));

    nn::Network network(std::move(layers), std::make_unique<nn::Adam>(0.01));

    std::println("Architecture:");
    for (std::size_t i = 0; i < network.size(); ++i) {
        std::println("  layer {}: {}", i, network.layer(i).name());
    }

    // Training data, column-wise: each column is one sample.
    const Eigen::MatrixXd batch_x{{0, 0, 1, 1},
                                  {0, 1, 0, 1}};
    const Eigen::MatrixXd batch_y{{0, 1, 1, 0}};

    // Evaluation set (all four XOR cases, with expected outputs).
    const Eigen::MatrixXd eval_x{{1, 1, 0, 0},
                                 {1, 0, 1, 0}};
    const Eigen::MatrixXd eval_y{{0, 1, 1, 0}};

    const int epochs = 3000;
    const int report_every = 300;

    std::println("\nTraining on XOR for {} epochs (Adam, lr = 0.01)...", epochs);
    for (int epoch = 0; epoch < epochs; ++epoch) {
        const Eigen::MatrixXd predicted = network.forward(batch_x);
        network.backward(batch_y, predicted);

        if (epoch % report_every == 0) {
            const double current_loss = nn::Network::loss(batch_y, predicted);
            std::println("  epoch {:>4} | loss = {:.6f}", epoch, current_loss);
        }
    }

    // Final evaluation, side by side with the expected values.
    const Eigen::MatrixXd final_pred = network.forward(eval_x);
    const double final_loss = nn::Network::loss(eval_y, final_pred);

    std::println("\nFinal evaluation (loss = {:.6f}):", final_loss);
    std::println("  {:>4} | {:>8} | {:>8}", "x", "expected", "predicted");
    for (int i = 0; i < eval_x.cols(); ++i) {
        std::println("  {:>1},{:<1} | {:>8.4f} | {:>8.4f}",
                     static_cast<int>(eval_x(0, i)), static_cast<int>(eval_x(1, i)),
                     eval_y(0, i), final_pred(0, i));
    }

    const bool converged = final_loss < 1e-3;
    std::println("\n{}", converged ? "XOR learned successfully."
                                   : "XOR did not converge; inspect seeds or lr.");
    return converged ? 0 : 1;
}
