#include <iostream>
#include <eigen/Dense>
#include <Layer.h>

int main() {
    // First, lets try creating a layer with 3 inputs and 2 outputs.
    Layer test_layer(3, 2);
    // We'll also need an input test.
    Eigen::MatrixXd test_input(3, 1);
    test_input << 0.2,
                  0.4,
                  0.9;

    // Now we try to do forward pass.
    Eigen::MatrixXd test_result = test_layer.forward(test_input);

    std::cout << "Test complete. Result of forward:\n" << test_result << std::endl;

    return 0;
}