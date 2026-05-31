#pragma once
#include <Eigen/Dense>
#include <vector>
//#include <unordered_map>
#include "Layer.hpp"

// Abstract or Generic Optimizer class:
class Optimizer {
    public:
        Optimizer(double learning_rate = 0.01) : learning_rate(learning_rate) {};
        virtual ~Optimizer() = default;
        
        // Add commentary here Sandy, pleaseeeee
        virtual void update(Layer & layer) = 0;
    protected:
        // All optimizers use this
        double learning_rate;
};

class SGD : public Optimizer {
    public:
        // Add commentary Sandy
        SGD(double learning_rate = 0.01) : Optimizer(learning_rate) {};

        // HEre too
        void update(Layer & layer) override;
};

/*

// Estructura para guardar el estado interno de Adam (Memoria)
struct AdamState {
    Eigen::MatrixXd m_w; // Momentum de los pesos
    Eigen::MatrixXd v_w; // Varianza de los pesos
    Eigen::VectorXd m_b; // Momentum de los sesgos
    Eigen::VectorXd v_b; // Varianza de los sesgos
};

// Optimizador Adam
class AdamOptimizer : public Optimizer {
private:
    double learning_rate;
    double beta1;
    double beta2;
    double epsilon;
    int t; // Contador de iteraciones (time step)

    // El mapa mágico: Clave = Puntero a la capa, Valor = Su memoria Adam
    std::unordered_map<Layer*, AdamState> states;

public:
    // Constructor con los valores recomendados por defecto en el paper de Adam
    AdamOptimizer(double lr = 0.001, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8);

    void step(std::vector<Layer>& layers) override;
};

*/
