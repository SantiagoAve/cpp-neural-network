/*
    OPTIMIZER & TRAINING TESTS:
    Closed-form checks of the SGD and Adam update rules, convergence on a linear-regression
    problem with a known solution, and XOR convergence with the same configuration used by
    the demo.
*/
#include "Activation.hpp"
#include "DenseLayer.hpp"
#include "Loss.hpp"
#include "Network.hpp"
#include "Optimizer.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {
    int failures = 0;

    void check(bool condition, const std::string & what) {
        if (!condition) {
            ++failures;
            std::cerr << "FAIL: " << what << '\n';
        }
    }

    void check_close(double a, double b, double tol, const std::string & what) {
        const double scale = std::max(1.0, std::max(std::abs(a), std::abs(b)));
        check(std::abs(a - b) <= tol * scale, what + " (a=" + std::to_string(a) + ", b="
                                 + std::to_string(b) + ")");
    }
} // (Anonymous namespace)

int main() {
    using namespace nn;

    // --- SGD: exact one-step update ---
    {
        DenseLayer layer(2, 2, Activation::identity(), 5u);
        const Eigen::MatrixXd weights_before = *layer.parameters().weights;
        const Eigen::VectorXd biases_before = *layer.parameters().biases;

        Eigen::MatrixXd input(2, 3);
        input << 1.0, 2.0, 3.0,
                 4.0, 5.0, 6.0;
        Eigen::MatrixXd targets(2, 3);
        targets << 0.5, -0.5, 1.0,
                   1.5, 0.5, -1.0;

        const Eigen::MatrixXd predicted = layer.forward(input);
        (void)layer.backward(Loss::mean_squared_derivative(targets, predicted));

        const Eigen::MatrixXd grad_w = *layer.parameters().grad_weights;
        const Eigen::VectorXd grad_b = *layer.parameters().grad_biases;

        SGD sgd(0.1);
        sgd.update(layer);

        const Eigen::MatrixXd expected_w = weights_before - 0.1 * grad_w;
        const Eigen::VectorXd expected_b = biases_before - 0.1 * grad_b;
        check((*layer.parameters().weights).isApprox(expected_w, 1e-12),
              "SGD weight update is exact");
        check((*layer.parameters().biases).isApprox(expected_b, 1e-12),
              "SGD bias update is exact");
    }

    // --- Adam: first-step closed form ---
    // After one update, m_hat = g and v_hat = g ^ 2, so the step is exactly:
    //     lr * g / (sqrt(g ^ 2) + eps)  ~  lr * sign(g).
    {
        DenseLayer layer(2, 2, Activation::identity(), 5u);
        const Eigen::MatrixXd weights_before = *layer.parameters().weights;
        const Eigen::VectorXd biases_before = *layer.parameters().biases;

        Eigen::MatrixXd input(2, 3);
        input << 1.0, 2.0, 3.0,
                 4.0, 5.0, 6.0;
        Eigen::MatrixXd targets(2, 3);
        targets << 0.5, -0.5, 1.0,
                   1.5, 0.5, -1.0;

        const Eigen::MatrixXd predicted = layer.forward(input);
        (void)layer.backward(Loss::mean_squared_derivative(targets, predicted));
        const Eigen::MatrixXd grad_w = *layer.parameters().grad_weights;
        const Eigen::VectorXd grad_b = *layer.parameters().grad_biases;

        Adam adam(0.01);
        adam.update(layer);

        const Eigen::MatrixXd expected_w =
            weights_before -
            0.01 * (grad_w.array() / (grad_w.array().square().sqrt() + 1e-8)).matrix();
        const Eigen::VectorXd expected_b =
            biases_before -
            0.01 * (grad_b.array() / (grad_b.array().square().sqrt() + 1e-8)).matrix();

        check((*layer.parameters().weights).isApprox(expected_w, 1e-12),
              "Adam first step (weights) matches closed form");
        check((*layer.parameters().biases).isApprox(expected_b, 1e-12),
              "Adam first step (biases) matches closed form");
    }

    // --- Linear regression with a known solution ---
    // y = 2x + 1, one Dense(1 -> 1) with identity activation. Adam must land on w = 2, b = 1.
    {
        Network network({}, std::make_unique<Adam>(0.05));
        network.add(std::make_unique<DenseLayer>(1, 1, Activation::identity(), 3u));

        Eigen::MatrixXd input(1, 8);
        input << -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0;
        Eigen::MatrixXd targets(1, 8);
        for (int i = 0; i < input.cols(); ++i) {
            targets(0, i) = 2.0 * input(0, i) + 1.0;
        }

        for (int step = 0; step < 600; ++step) {
            const Eigen::MatrixXd predicted = network.forward(input);
            network.backward(targets, predicted);
        }

        const Eigen::MatrixXd predicted = network.forward(input);
        const double loss = Network::loss(targets, predicted);
        check(loss < 1e-4, "linear regression converged (loss < 1e-4)");
        check_close(predicted(0, 7), 5.0, 1e-2, "predicts y(2) = 5");
    }

    // --- XOR convergence (Same configuration as the demo) ---
    // NOTE on seeds: with ReLU hidden units an unlucky initialization can leave every hidden
    // neuron dead (z < 0 for all four inputs), in which case XOR never leaves 0.5. Seeds are
    // fixed so the test is stable; a wider hidden layer or a different seed fixes it in practice.
    {
        std::vector<std::unique_ptr<Layer>> layers;
        layers.push_back(std::make_unique<DenseLayer>(2, 4, Activation::relu(), 4u));
        layers.push_back(std::make_unique<DenseLayer>(4, 1, Activation::sigmoid(), 104u));
        Network network(std::move(layers), std::make_unique<Adam>(0.01));

        const Eigen::MatrixXd batch_x{{0, 0, 1, 1}, {0, 1, 0, 1}};
        const Eigen::MatrixXd batch_y{{0, 1, 1, 0}};

        for (int epoch = 0; epoch < 3000; ++epoch) {
            const Eigen::MatrixXd predicted = network.forward(batch_x);
            network.backward(batch_y, predicted);
        }

        const Eigen::MatrixXd predicted = network.forward(batch_x);
        const double loss = Network::loss(batch_y, predicted);
        check(loss < 1e-3, "XOR converged (loss < 1e-3 after 3000 epochs)");

        for (int i = 0; i < 4; ++i) {
            check_close(predicted(0, i), batch_y(0, i), 0.05, "XOR output");
        }
    }

    if (failures == 0) {
        std::cout << "All optimizer/training checks passed.\n";
        return EXIT_SUCCESS;
    }
    std::cerr << failures << " check(s) failed.\n";
    return EXIT_FAILURE;
}
