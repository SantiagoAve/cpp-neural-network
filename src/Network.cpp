#include "Network.hpp"

/*
    FORWARD & BACKWARD:
*/
Eigen::MatrixXd Network::net_forward(const Eigen::MatrixXd & input) {
    Eigen::MatrixXd cur_output = input; // This works as an iterator.

    for (auto & layer : layers) {
        cur_output = layer.forward(cur_output);
    }
    
    return cur_output;
}

void Network::net_backward(const Eigen::MatrixXd & true_values,
                           const Eigen::MatrixXd & predicted_values) {
    // This is the initial '2 * (a^L - y^L)' that you see normally in calculus.
    // From here, you backpropagate.
    Eigen::MatrixXd propag_loss_grad = Loss::mean_derivative(true_values, predicted_values);

    // Using a size_t is preffereable, but it caused many runtimeErrors, since it overflowed.
    for (int i = static_cast<int>(this->num_layers) - 1; i >= 0; i--) {
        propag_loss_grad = this->layers[i].backward(propag_loss_grad);
    }

    // Then you do the same for the optimization:
    for (int i = static_cast<int>(this->num_layers) - 1; i >= 0; i--) {
        this->optimizer->update(this->layers[i]);
    }
}
/*
    -----------------------------------------------------------------------------------------------
*/