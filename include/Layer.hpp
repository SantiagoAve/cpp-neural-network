#pragma once
#include <Eigen/Dense>
#include <string>
#include <functional>

class Layer {
    // This is a powerful alias, used so the Layer can store it's own activation function.
    using ActivationFunction = std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)>;
    
    public:
        /*
            LAYER CONSTRUCTOR:
            Initializes a Layer. Creates a weight matrix of size Output*Input, creates
            a bias matrix of side Output and uses one of two activations.
        */
        Layer(int input_size, int output_size, ActivationFunction activation_function,
              ActivationFunction activation_derivative);

        /*
            FORWARD ACTION:
            Funcion that returns an output matrix. Said matrix is of size Output*Bach,
            were each component is the specific activation of a Neuron. Bach ...
        */
        Eigen::MatrixXd forward(const Eigen::MatrixXd & input);

        /*
            BACKWARD ACTION:
            Function that updates weights and biases to a more fitting value. Uses the
            gradient descent principle, so it uses chain rule of the error formula.
        */
        Eigen::MatrixXd backward(const Eigen::MatrixXd & propag_loss_grad, double learning_rate);

    private:
        /*
            INNER VARIABLES:
            These variables are used to store information of the layer. They are mostly
            used for the backward function.
        */
        Eigen::MatrixXd weights; // Member' Output*Input in size.
        Eigen::VectorXd biases;

        Eigen::MatrixXd derivative_w;
        Eigen::MatrixXd derivative_b;

        ActivationFunction activation_function; // Can be Sigmoid, ReLU, TanH.
        ActivationFunction activation_derivative;

        Eigen::MatrixXd layer_z; // Pre-activation matrix value.
        Eigen::MatrixXd prev_input;
};