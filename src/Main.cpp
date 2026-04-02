#include <iostream>
#include <eigen/Dense>
#include <Layer.h>

int main() {
    // First, lets try creating a layer with 3 inputs and 2 outputs, with sigmoid.
    Layer test_layer_sigmoid(3, 2, "Sigmoid");
    // Then, create another layer with 3 inputs and 2 outputs, but with relu.
    Layer test_layer_relu(3, 2, "ReLU");
    // We'll also need an input test for both.
    Eigen::MatrixXd test_input(3, 1);
    test_input << 0.2,
                  0.4,
                  0.9;

    // Now we try to do forward pass in sigmoid.
    Eigen::MatrixXd test_result_sigmoid = test_layer_sigmoid.forward(test_input);
    // Then we try to do forward pass in relu.
    Eigen::MatrixXd test_result_relu = test_layer_relu.forward(test_input);

    // Print each case and see it youself!
    std::cout << "Test complete. Result of forward in Sigmoid:\n" << test_result_sigmoid << std::endl;
    std::cout << "Test complete. Result of forward in ReLU:\n" << test_result_relu << std::endl;

    return 0;
}