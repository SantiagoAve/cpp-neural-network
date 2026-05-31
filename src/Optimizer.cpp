/*
    REVIEW  REVIEW  REVIEW  REVIEW  REVIEW  REVIEW  REVIEW  REVIEW  REVIEW  REVIEW  REVIEW  REVIEW  REVIEW  REVIEW  REVIEW
    REVIEW  REVIEW  REVIEW  REVIEW  REVIEW  REVIEW  REVIEW  REVIEW  REVIEW  REVIEW  REVIEW  REVIEW  REVIEW  REVIEW  REVIEW
    REVIEW  REVIEW  REVIEW  REVIEW  REVIEW  REVIEW  REVIEW  REVIEW  REVIEW  REVIEW  REVIEW  REVIEW  REVIEW  REVIEW  REVIEW
    REVIEW  REVIEW  REVIEW  REVIEW  REVIEW  REVIEW  REVIEW  REVIEW  REVIEW  REVIEW  REVIEW  REVIEW  REVIEW  REVIEW  REVIEW
*/
#include "Optimizer.hpp"

// Implementación del step
void SGD::update(Layer & layer) {
    // 1. Obtenemos las referencias a los parámetros reales y sus gradientes
    Eigen::MatrixXd& w = layer.get_weights();
    Eigen::VectorXd& b = layer.get_biases();
    Eigen::MatrixXd& dw = layer.get_derivative_w();
    Eigen::VectorXd& db = layer.get_derivative_b();

    // 2. Aplicamos la regla de actualización de SGD
    // Como son referencias, esto modifica directamente las matrices dentro de la Layer
    w -= learning_rate * dw;
    b -= learning_rate * db;
}