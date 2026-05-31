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
            Initializes a Layer. Creates a weight matrix of size Output*Input, creates a
            bias matrix of side Output and loads activation functions and its derivative.
        */
        Layer(int input_size, int output_size, ActivationFunction activation_function,
              ActivationFunction activation_derivative);

        /*
            FORWARD ACTION:
            Function that returns an output matrix of size Output*Bach. Each component
            would be the specific activation of a single Neuron. Bach is optional, but
            highly recommended to be more efficient.
        */
        Eigen::MatrixXd forward(const Eigen::MatrixXd & input);

        /*
            BACKWARD ACTION:
            Function that propagates the gradient of the loss function. This doesn't
            update any weights nor biases. That's the Optimizer's job. This uses the
            gradient descent principle, so you use the chain rule for the error.
        */
        Eigen::MatrixXd backward(const Eigen::MatrixXd & propag_loss_grad);

        /*
            GETTERS:
            These are getter functions, they are very basic and are used by the Optimizer
            to quickly access to layer's information.
        */
        Eigen::MatrixXd & get_weights() { return this->weights; }
        Eigen::VectorXd & get_biases() { return this->biases; }
        Eigen::MatrixXd & get_derivative_w() { return this->derivative_w; }
        Eigen::VectorXd & get_derivative_b() { return this->derivative_b; }

    private:
        /*
            INNER VARIABLES:
            These variables are used to store information of the layer. They are mostly
            used for the backward function.
        */
        Eigen::MatrixXd weights; // Member' Output*Input in size.
        Eigen::VectorXd biases;

        Eigen::MatrixXd derivative_w;
        Eigen::VectorXd derivative_b;

        ActivationFunction activation_function; // Can be Sigmoid, ReLU, TanH.
        ActivationFunction activation_derivative;

        Eigen::MatrixXd layer_z; // Pre-activation matrix value.
        Eigen::MatrixXd prev_input;
};