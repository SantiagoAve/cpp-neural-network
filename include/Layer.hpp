#pragma once
#include <Eigen/Dense>
#include <string>
#include "ActivationF.hpp"

class Layer {
    public:
        /*
            LAYER CONSTRUCTOR:
            Initializes a Layer. Creates a weight matrix of size Output*Input, creates
            a bias matrix of side Output and uses one of two activations.
        */
        Layer(int input_size, int output_size, const std::string & activation = "Sigmoid");

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
            used for backward, like my_z and my_a or prev_input.
        */
        Eigen::MatrixXd weights; // Member' Output*Input in size.
        Eigen::VectorXd biases;
        Eigen::MatrixXd layer_z; // Pre-activation matrix value.
        Eigen::MatrixXd layer_a;
        Eigen::MatrixXd prev_input;
        std::string activation_function; // Can be Sigmoid, ReLU, TanH.
};