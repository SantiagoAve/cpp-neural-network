#include "Optimizer.hpp"

#include <cmath>
#include <format>
#include <stdexcept>

namespace nn {
    /*
        OPTIMIZER (ABSTRACT):
        This is the constructor for the abstract base class. It simply sets the learning_rate.
    */
    Optimizer::Optimizer(double learning_rate) : learning_rate_(learning_rate) {
        if (learning_rate_ <= 0.0) {
            throw std::invalid_argument(
                std::format("Optimizer: learning rate must be positive (got {})",
                            learning_rate_));
        }
    }
    /*
        ---------------------------------------------------------------------------
    */

    // The SGD constructor is a call to the base one.
    SGD::SGD(double learning_rate) : Optimizer(learning_rate) {}

    /*
        SGD UPDATE:
        This mutates the layers in place, using the references to the real tensors
        stored in the parameters() view.

    */
    void SGD::update(Layer & layer) {
        Parameters p = layer.parameters();
        if (p.empty()) {
            return; // Parameterless layers? Nothing to update.
        }

        *p.weights -= learning_rate_ * *p.grad_weights;
        *p.biases -= learning_rate_ * *p.grad_biases;
    }
    /*
        ---------------------------------------------------------------------------
    */

    // The ADAM constructor. It needs some extra checking on its variables.
    Adam::Adam(double learning_rate, double beta1, double beta2, double epsilon)
        : Optimizer(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon) {
        if (beta1_ <= 0.0 || beta1_ >= 1.0 || beta2_ <= 0.0 || beta2_ >= 1.0) {
            throw std::invalid_argument(
                std::format("Adam: beta values must be in (0, 1) (got beta1={}, beta2={})",
                            beta1_, beta2_));
        }
        if (epsilon_ <= 0.0) {
            throw std::invalid_argument(
                std::format("Adam: epsilon must be positive (got {})", epsilon_));
        }
    }

    /*
        ADAM UPDATE:
        The first time a layer is seen, the State (Momoment estimates and per-layer step count)
        is lazily generated, sized to that layer's parameters.
    */
    void Adam::update(Layer & layer) {
        Parameters p = layer.parameters();
        if (p.empty()) {
            return; // Parameterless layers? Again, nothing to update.
        }

        State & s = state_[&layer];
        if (s.m_w.size() == 0) {
            s.m_w = Eigen::MatrixXd::Zero(p.weights->rows(), p.weights->cols());
            s.v_w = Eigen::MatrixXd::Zero(p.weights->rows(), p.weights->cols());
            s.m_b = Eigen::VectorXd::Zero(p.biases->size());
            s.v_b = Eigen::VectorXd::Zero(p.biases->size());
        }
        ++s.t;

        // --- Weights ---
        s.m_w = beta1_ * s.m_w + (1.0 - beta1_) * *p.grad_weights;
        s.v_w = beta2_ * s.v_w + (1.0 - beta2_) * (*p.grad_weights).array().square().matrix();

        const double bias1 = 1.0 - std::pow(beta1_, static_cast<double>(s.t));
        const double bias2 = 1.0 - std::pow(beta2_, static_cast<double>(s.t));

        const Eigen::MatrixXd m_hat_w = s.m_w / bias1;
        const Eigen::MatrixXd v_hat_w = s.v_w / bias2;

        *p.weights -= learning_rate_ *
                    (m_hat_w.array() / (v_hat_w.array().sqrt() + epsilon_)).matrix();

        // --- Biases ---
        s.m_b = beta1_ * s.m_b + (1.0 - beta1_) * *p.grad_biases;
        s.v_b = beta2_ * s.v_b + (1.0 - beta2_) * (*p.grad_biases).array().square().matrix();

        const Eigen::VectorXd m_hat_b = s.m_b / bias1;
        const Eigen::VectorXd v_hat_b = s.v_b / bias2;

        *p.biases -= learning_rate_ *
                    (m_hat_b.array() / (v_hat_b.array().sqrt() + epsilon_)).matrix();
    }
} // namespace nn
