/*
    GRADIENT CHECK (the gold standard for backprop):
    Compares every analytic gradient produced by Layer::backward against a central
    finite-difference approximation of the true loss. If backprop implements the chain
    rule correctly, the relative error must be at the rounding-error level (~1e-9 or
    better in double precision).

    Checked for every weight, every bias and the network input, on two architectures
    (tanh -> sigmoid and a pure linear layer).
*/
#include "Activation.hpp"
#include "DenseLayer.hpp"
#include "Loss.hpp"
#include "Network.hpp"
#include "Optimizer.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {
    int failures = 0;
    double max_relative_error = 0.0;

    void check(bool condition, const std::string & what) {
        if (!condition) {
            ++failures;
            std::cerr << "FAIL: " << what << '\n';
        }
    }

    void record_error(double analytic, double numeric) {
        const double denom = std::max({std::abs(analytic), std::abs(numeric), 1e-3});
        max_relative_error = std::max(max_relative_error, std::abs(analytic - numeric) / denom);
    }

    // Full forward through the layer stack (recomputes everything; caches get refreshed,
    // which is fine because analytic gradients were stored already).
    Eigen::MatrixXd full_forward(std::vector<nn::DenseLayer*> & stack, const Eigen::MatrixXd & input) {
        Eigen::MatrixXd out = input;
        for (auto * layer : stack) {
            out = layer->forward(out);
        }
        return out;
    }

    // Runs one forward + backward pass and returns the loss value.
    double run_pass(std::vector<nn::DenseLayer*> & stack, const Eigen::MatrixXd & input,
                    const Eigen::MatrixXd & targets) {
        const Eigen::MatrixXd predicted = full_forward(stack, input);
        Eigen::MatrixXd gradient = nn::Loss::mean_squared_derivative(targets, predicted);
        for (auto it = stack.rbegin(); it != stack.rend(); ++it) {
            gradient = (*it)->backward(gradient);
        }
        return nn::Loss::mean_squared(targets, predicted);
    }

    void check_layer_gradients(std::vector<nn::DenseLayer*> & stack, const Eigen::MatrixXd & input,
                            const Eigen::MatrixXd & targets, const std::string & tag, double tolerance) {
        const double base_loss_before = run_pass(stack, input, targets);

        // Snapshot analytic gradients (backward already stored them).
        std::vector<Eigen::MatrixXd> grad_w;
        std::vector<Eigen::VectorXd> grad_b;
        for (auto * layer : stack) {
            grad_w.push_back(layer->parameters().grad_weights->eval());
            grad_b.push_back(layer->parameters().grad_biases->eval());
        }

        const double eps = 1e-6;

        // --- Weights ---
        for (std::size_t l = 0; l < stack.size(); ++l) {
            Eigen::MatrixXd & w = *stack[l]->parameters().weights;
            for (int i = 0; i < w.rows(); ++i) {
                for (int j = 0; j < w.cols(); ++j) {
                    const double original = w(i, j);

                    w(i, j) = original + eps;
                    const double loss_plus = nn::Loss::mean_squared(targets, full_forward(stack, input));

                    w(i, j) = original - eps;
                    const double loss_minus = nn::Loss::mean_squared(targets, full_forward(stack, input));

                    w(i, j) = original; // Exact restore.
                    const double numeric = (loss_plus - loss_minus) / (2.0 * eps);
                    record_error(grad_w[l](i, j), numeric);
                    check(std::abs(grad_w[l](i, j) - numeric) <= tolerance * std::max(1.0, std::abs(numeric)),
                          tag + ": weight gradient (" + std::to_string(l) + "," + std::to_string(i) + ","
                          + std::to_string(j) + ")");
                }
            }
        }

        // --- Biases ---
        for (std::size_t l = 0; l < stack.size(); ++l) {
            Eigen::VectorXd & b = *stack[l]->parameters().biases;
            for (int i = 0; i < b.size(); ++i) {
                const double original = b(i);

                b(i) = original + eps;
                const double loss_plus = nn::Loss::mean_squared(targets, full_forward(stack, input));

                b(i) = original - eps;
                const double loss_minus = nn::Loss::mean_squared(targets, full_forward(stack, input));

                b(i) = original;
                const double numeric = (loss_plus - loss_minus) / (2.0 * eps);
                record_error(grad_b[l](i), numeric);
                check(std::abs(grad_b[l](i) - numeric) <= tolerance * std::max(1.0, std::abs(numeric)),
                      tag + ": bias gradient (" + std::to_string(l) + "," + std::to_string(i) + ")");
            }
        }

        // --- Input gradient (feeds future stacked layers, e.g. convolutional) ---
        run_pass(stack, input, targets); // Refresh caches and re-run backward.
        Eigen::MatrixXd input_gradient = input;
        {
            Eigen::MatrixXd gradient(targets.rows(), targets.cols());
            gradient.setZero();
            const Eigen::MatrixXd predicted = full_forward(stack, input);
            gradient = nn::Loss::mean_squared_derivative(targets, predicted);
            for (auto it = stack.rbegin(); it != stack.rend(); ++it) {
                gradient = (*it)->backward(gradient);
            }
            input_gradient = gradient;
        }
        Eigen::MatrixXd x = input;
        for (int i = 0; i < x.rows(); ++i) {
            for (int j = 0; j < x.cols(); ++j) {
                const double original = x(i, j);
                x(i, j) = original + eps;
                const double loss_plus =
                    nn::Loss::mean_squared(targets, full_forward(stack, x));
                x(i, j) = original - eps;
                const double loss_minus =
                    nn::Loss::mean_squared(targets, full_forward(stack, x));
                x(i, j) = original;
                const double numeric = (loss_plus - loss_minus) / (2.0 * eps);
                record_error(input_gradient(i, j), numeric);
                check(std::abs(input_gradient(i, j) - numeric) <=
                        tolerance * std::max(1.0, std::abs(numeric)),
                    tag + ": input gradient (" + std::to_string(i) + "," + std::to_string(j) + ")");
            }
        }

        // Loss must be unchanged after all perturbations were restored.
        run_pass(stack, input, targets);
        check(std::abs(run_pass(stack, input, targets) - base_loss_before) < 1e-12,
              tag + ": loss unchanged after gradient restoration");
    }
} // (Anonymous namespace)

int main() {
    using namespace nn;

    // --- Architecture A: 3 -> 5 (tanh) -> 2 (sigmoid) ---
    {
        DenseLayer hidden(3, 5, Activation::tanh(), 42u);
        DenseLayer output(5, 2, Activation::sigmoid(), 1337u);
        std::vector<DenseLayer *> stack{&hidden, &output};

        Eigen::MatrixXd input(3, 7);
        input << 0.30, -0.90, 0.10, 1.20, -0.40, 0.75, -1.10,
                 0.85, 0.20, -0.35, 0.05, 1.40, -0.60, 0.15,
                 -0.50, 0.65, 0.95, -1.25, 0.00, 0.45, -0.20;

        Eigen::MatrixXd targets(2, 7);
        targets << 0.10, 0.90, 0.40, 0.75, 0.20, 0.60, 0.95,
                   0.80, 0.15, 0.55, 0.30, 0.85, 0.05, 0.45;

        check_layer_gradients(stack, input, targets, "tanh->sigmoid", 1e-6);
    }

    // --- Architecture B: single linear layer (regression shape) ---
    {
        DenseLayer output(3, 2, Activation::identity(), 2024u);
        std::vector<DenseLayer *> stack{&output};

        Eigen::MatrixXd input(3, 5);
        input << 1.0, 2.0, 3.0, 4.0, 5.0,
                 0.5, -1.0, 1.5, -2.0, 2.5,
                 -0.3, 0.7, -1.1, 0.9, -0.5;

        Eigen::MatrixXd targets(2, 5);
        targets << 0.1, -0.4, 0.9, -1.2, 1.6,
                   0.7, 0.2, -0.8, 1.1, -0.3;

        check_layer_gradients(stack, input, targets, "identity", 1e-6);
    }

    // --- Network orchestration ------------------------------------------------
    {
        std::vector<std::unique_ptr<Layer>> layers;
        auto l0 = std::make_unique<DenseLayer>(3, 5, Activation::tanh(), 42u);
        auto l1 = std::make_unique<DenseLayer>(5, 2, Activation::sigmoid(), 1337u);
        DenseLayer * raw0 = l0.get();
        DenseLayer * raw1 = l1.get();
        layers.push_back(std::move(l0));
        layers.push_back(std::move(l1));
        Network network(std::move(layers), std::make_unique<SGD>(0.1));

        Eigen::MatrixXd input(3, 7);
        input << 0.30, -0.90, 0.10, 1.20, -0.40, 0.75, -1.10,
                 0.85, 0.20, -0.35, 0.05, 1.40, -0.60, 0.15,
                 -0.50, 0.65, 0.95, -1.25, 0.00, 0.45, -0.20;
        Eigen::MatrixXd targets(2, 7);
        targets << 0.10, 0.90, 0.40, 0.75, 0.20, 0.60, 0.95,
                   0.80, 0.15, 0.55, 0.30, 0.85, 0.05, 0.45;

        // Network::forward must equal the manual chain.
        const Eigen::MatrixXd via_network = network.forward(input);
        const Eigen::MatrixXd manual = raw1->forward(raw0->forward(input));
        check(via_network.isApprox(manual, 1e-12), "Network::forward == manual chain");

        // backward() must actually update the parameters through the optimizer.
        const Eigen::MatrixXd before = *raw0->parameters().weights;
        network.backward(targets, network.forward(input));
        const Eigen::MatrixXd after = *raw0->parameters().weights;
        check(!before.isApprox(after), "Network::backward updates parameters");
    }

    std::cout << "Max relative gradient error: " << max_relative_error << '\n';
    check(max_relative_error < 1e-6, "gradient check tolerance");

    if (failures == 0) {
        std::cout << "All gradient checks passed.\n";
        return EXIT_SUCCESS;
    }
    std::cerr << failures << " check(s) failed.\n";
    return EXIT_FAILURE;
}
