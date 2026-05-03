#include <iostream>
#include <eigen/Dense>
#include "Layer.hpp"
#include "Network.hpp"
// TEST ONLY:
#include "Loss.hpp"

/*
    RIGHT NOW, WE'LL DO A SIMPLE TEST TO CHECK IF THIS NEURAL NETWORK IS WORKING
    AS INTENDED. THE IDEA IS TO TRAIN THIS NN TO SIMULATE A XOR.
*/
int main() {
    // For this, our structure is: { 2 -> 4 -> 1 }
    Layer layer_1(2, 4, "ReLU");
    Layer layer_2(4, 1, "Sigmoid");
    Network network({layer_1, layer_2});

    // Let's create four simple examples in a 2*4 matrix:
    Eigen::MatrixXd test_input(2, 4);
    test_input << 0, 0, 1, 1,
                  0, 1, 0, 1;

    // Then, our real values are:
    Eigen::MatrixXd test_true_val(1, 4);
    test_true_val << 0, 1, 1, 0;

    // Now we setup out training configuration:
    double test_lear_rate = 0.1;
    short epochs = 10000; // An epoch is a single complete pass through the training dataset.

    for (short i = 0; i < epochs; i++) {
        // First, do forward:
        Eigen::MatrixXd test_pred_values = network.net_forward(test_input);

        // Afterwards, proceed with backward:
        network.net_backward(test_true_val, test_pred_values, test_lear_rate);

        // Lastly, we print some info after many epochs:
        if (i % 1000 == 0) {
            // Calculate and print current loss:
            double loss = Loss::mean_squared(test_true_val, test_pred_values);
            std::cout << "This is Epoch n°: " << i << " | Current loss at: " << loss << std::endl;
        }
    }

    // These are the final results:
    Eigen::MatrixXd test_final_pred = network.net_forward(test_input);
    std::cout << "\n" << "After " << epochs << " epochs, the final preficted values are..." << std::endl
              << "\n" << test_final_pred << std::endl;

    return 0;
}