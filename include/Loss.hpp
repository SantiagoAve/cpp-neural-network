#pragma once

#include <Eigen/Dense>
#include <string>

namespace nn {
    /*
        LOSS TYPE:
        Which loss a Network uses for both training and reporting. The network stores one of
        these at construction and both backward() and loss() dispatch on it.
    */
    enum class LossType { MeanSquared, SoftmaxCrossEntropy };

    /*
        LOSS TAGS:
        Text form of a LossType ("mse" / "cross_entropy"), which is what gets written
        into saved model files. parse() rejects anything unknown.
    */
    [[nodiscard]] const char * loss_type_tag(LossType type);
    [[nodiscard]] LossType parse_loss_type(const std::string & tag);

    /*
        LOSS FUNCTIONS:
        These functions measure how wrong the network predictions are; a bigger number means a
        worse prediction. Its gradient with respect to the prediction is the seed for backprop,
        as it tells you the direction of maximum growth. We want the opposite way, of course.

        OPTIONS:
        1. "MSE" (Mean Squared Error).
        2. "Cross-Entropy" (Softmax Cross-Entropy).

        A NOTE ON CROSS-ENTROPY:
        It expects RAW SCORES (logits), not probabilities: use Identity as the output layer's
        activation and let the loss apply softmax internally (The same design as PyTorch's
        CrossEntropyLoss). Fusing them is what makes the gradient so clean, (p - y) / N, and the
        log-softmax trick is what keeps it numerically stable. Do NOT put a SoftmaxLayer as the
        last layer AND select cross_entropy; that would softmax twice.
    */
    class Loss {
        public:
            /*
                MSE:
                This is the MSE averaged over all matrix elements, so it's:

                    L(Y, P) = mean((Y - P) ^ 2)

                Also, throws an exception, std::invalid_argument, when shapes differ.
            */
            [[nodiscard]] static double mean_squared(const Eigen::MatrixXd & true_values,
                                                     const Eigen::MatrixXd & predicted_values);
            
            /*
                MSE GRADIENT:
                This is the gradient of the MSE with respect to the predictions, so it's:

                    dL/dP = 2 * (P - Y) / element_count

                You divide by the element count, not just the batch size, so this works fine
                for any output dimensionality.
            */
            [[nodiscard]] static Eigen::MatrixXd mean_squared_derivative(const Eigen::MatrixXd & true_values,
                                                                         const Eigen::MatrixXd & predicted_values);

            /*
                CROSS-ENTROPY (SOFTMAX FUSED):
                Given logits Z (The network's raw scores) and targets Y (One-hot, or any valid
                distribution over classes, one column per sample):

                    L = -(1 / N) * sum over samples of sum over classes of Y * log(softmax(Z))

                Written with log-softmax for stability, this never evaluates log(0) from
                underflow and never overflows exp(), no matter how extreme the scores are.
                With uniform logits the value is exactly log(K), a nice sanity check.
            */
            [[nodiscard]] static double cross_entropy(const Eigen::MatrixXd & true_values,
                                                      const Eigen::MatrixXd & logits);

            /*
                CROSS-ENTROPY GRADIENT:
                The whole point of the fused design; the gradient with respect to the logits is:

                    dL/dZ = (softmax(Z) - Y) / N

                With N being the batch size (columns). Simple, exact, and stable; computing this
                through softmax and log separately would be slower and would underflow.
            */
            [[nodiscard]] static Eigen::MatrixXd cross_entropy_derivative(const Eigen::MatrixXd & true_values,
                                                                          const Eigen::MatrixXd & logits);
    }; // class Loss
} // namespace nn
