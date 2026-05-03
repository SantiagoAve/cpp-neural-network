#include <iostream>
#include <eigen/Dense>
#include "Layer.hpp"
#include "Network.hpp"
#include "ActivationF.hpp"
// TEST ONLY:
#include "Loss.hpp"

/*
    RIGHT NOW, WE'LL DO A SIMPLE TEST TO CHECK IF THIS NEURAL NETWORK IS WORKING
    AS INTENDED. THE IDEA IS TO TRAIN THIS NN TO SIMULATE A XOR.
*/
int main() {
    // For this, our structure is: { 2 -> 4 -> 1 }
    Layer layer_1(2, 4, ActivationFunction::relu, ActivationFunction::relu_derivative);
    Layer layer_2(4, 1, ActivationFunction::sigmoid, ActivationFunction::sigmoid_derivative);
    Network network({layer_1, layer_2});

    // Let's do one bach:
    Eigen::MatrixXd test_input_1(2, 4);
    test_input_1 << 0, 0, 1, 1,
                    0, 1, 0, 1;

    Eigen::MatrixXd test_true_val_1(1, 4);
    test_true_val_1 << 0, 1, 1, 0;

    // Now, another one:
    Eigen::MatrixXd test_input_2(2, 4);
    test_input_2 << 1, 1, 0, 0,
                    0, 1, 1, 0;
    
    Eigen::MatrixXd test_true_val_2(1, 4);
    test_true_val_2 << 1, 0, 1, 0;

    // Finally, an example not seen:
    Eigen::MatrixXd test_input_3(2, 4);
    test_input_3 << 1, 1, 0, 0,
                    1, 0, 1, 0;

    // Now we setup out training configuration:
    double test_lear_rate = 0.1;
    short epochs = 10000; // An epoch is a single complete pass through the training dataset.

    for (short i = 0; i < epochs; i++) {
        
        // First, do one full bach:
        Eigen::MatrixXd test_pred_values_1 = network.net_forward(test_input_1);

        network.net_backward(test_true_val_1, test_pred_values_1, test_lear_rate);

        // Now, with another bach:
        Eigen::MatrixXd test_pred_values_2 = network.net_forward(test_input_2);

        network.net_backward(test_true_val_2, test_pred_values_2, test_lear_rate);

        // Lastly, we print some info after many epochs:
        if (i % 1000 == 0) {
            // Calculate and print current loss:
            double loss = Loss::mean_squared((test_true_val_1 + test_true_val_2) / 2,
                                             (test_pred_values_1 + test_pred_values_2) / 2);
            std::cout << "This is Epoch n°: " << i << " | Current loss at: " << loss << std::endl;
        }
    }

    // These are the final results:
    Eigen::MatrixXd test_final_pred = network.net_forward(test_input_3);
    std::cout << "\n" << "After " << epochs << " epochs, the final preficted values are..." << std::endl
              << "\n" << test_final_pred << std::endl;

    return 0;
}