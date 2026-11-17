#pragma once

#include "Layer.hpp"

namespace nn {
    /*
        SOFTMAX:
        Turns a column of raw scores (Logits) into a probability distribution over the features:
        every entry ends up in (0, 1) and each column sums to exactly 1.

        Numerical note: the naive formula exp(z_i) / sum(exp(z)) overflows as soon as a score is
        moderately large. The stable version subtracts the column maximum first (Softmax is
        shift-invariant, so this changes nothing mathematically) and only then exponentiates,
        keeping every exponent in (-inf, 0]. This is the same trick every serious framework uses.
    */
    [[nodiscard]] Eigen::MatrixXd softmax(const Eigen::MatrixXd & logits);

    /*
        SOFTMAX LAYER:
        A parameterless layer that applies softmax to every column of its input. Because softmax
        couples all the features of a column, its backward pass is not an element-wise product:
        it is the full vector-Jacobian product

            dL/dz_i = p_i * (g_i - sum_j(g_j * p_j))

        which this layer computes exactly, so it can sit anywhere in the stack (An attention block
        later in the roadmap will reuse exactly this math).

        Since it owns no tensors, parameters() returns an empty view and every optimizer simply
        skips it, and save() only needs to record its type.
    */
    class SoftmaxLayer final : public Layer {
        public:
            Eigen::MatrixXd forward(const Eigen::MatrixXd & input) override;
            Eigen::MatrixXd backward(const Eigen::MatrixXd & output_gradient) override;

            void save(std::ostream & out) const override;

            [[nodiscard]] std::string name() const override;

        private:
            Eigen::MatrixXd probabilities_; // Cached from forward(); backward needs it.
    }; // class SoftmaxLayer
} // namespace nn
