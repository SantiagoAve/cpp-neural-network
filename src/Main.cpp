#include <iostream>
#include <eigen/Dense>
#include "Layer.h"
#include "Network.h"
// TEST ONLY:
#include "Loss.h"

int main() {
    // Let's create the first layer, which uses ReLU. Then, we
    // move to the second one using Sigmoid.
    Layer layer_1(10, 5, "ReLU");
    Layer layer_2(5, 2, "Sigmoid");

    // Now we can create the Network.
    Network network({layer_1, layer_2});

    // Let's create the fake input.
    Eigen::MatrixXd test_input(10, 1);
    test_input << 0.25,
                  0.40,
                  0.91,
                  0.89,
                  0.50,
                  0.21,
                  0.37,
                  0.11,
                  0.05,
                  1.00;

    // Now we make the full Network forward.
    Eigen::MatrixXd test_predict = network.net_forward(test_input);

    // Print the final result.
    std::cout << "Test complete. Result of network forward:\n" << test_predict << std::endl;

     // And now we make a test for Loss function.
    Eigen::MatrixXd test_trueval(2, 1);
    test_trueval << 1.0,
                    0.0;

    double test_loss = Loss::mean_squared(test_trueval, test_predict);
    std::cout << "Loss registered: " << test_loss << std::endl;

    Eigen::MatrixXd test_gradloss = Loss::mean_derivative(test_trueval, test_predict);
    std::cout << "Loss gradient:\n" << test_gradloss << std::endl;

    return 0;
}