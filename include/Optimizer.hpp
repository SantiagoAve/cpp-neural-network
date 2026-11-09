#pragma once
#include "Layer.hpp"
#include <unordered_map>

namespace nn {
    /*
        OPTIMIZER (ABSTRACT):
        A strategy pattern that turns the gradients stored by each layer into parameter updates.
        Netowork calls update() once per layer after every backward pass, so any subclasses only
        need to describe what one step means.

        Parameterless layers are skipped automatically, which keeps some layers free of optimizer
        concerns.
    */
    class Optimizer {
        public:
            explicit Optimizer(double learning_rate = 0.01);
            virtual ~Optimizer() = default;

            /*
                UPDATE:
                This function updates network parameters, such as weights and biases, allowing
                a smaller error in the next passes.
            */
            virtual void update(Layer& layer) = 0;

            // Simple learning_rate getter.
            [[nodiscard]] double learning_rate() const { return learning_rate_; }

        protected:
            double learning_rate_;
    };

    /*
        STOCHASTIC GRADIENT DESCENT (plain):
        The classical update rule: W = W - lr * dL/dW. Simple, robust, and the baseline every
        other optimizer must beat.
    */
    class SGD : public Optimizer {
        public:
            explicit SGD(double learning_rate = 0.01);
    
            void update(Layer & layer) override;
    };
    
    /*
        ADAM:
        Or Adaptive Moment Estimation. Keeps per-parameter exponential moving averages of the
        gradient (m, the first moment, "momentum") and of its square (v, the second moment, or
        "variance"), then corrects both for their zero-initialization bias before stepping:
    
        m_t = beta1 * m_(t-1) + (1 - beta1) * g_t
        v_t = beta2 * v_(t-1) + (1 - beta2) * g_t ^ 2
        m'_t = m_t / (1 - beta1 ^ t) -> bias correction
        v'_t = v_t / (1 - beta2 ^ t)
        w = w - lr * m'_t / (sqrt(v'_t) + epsilon)
    
        States are kept per layer (lazy-initialized on first update). Keys are stable because
        Network stores layers as unique_ptr, so the pointed-to objects never move.
    */
    class Adam : public Optimizer {
        public:
            explicit Adam(double learning_rate = 1e-3, double beta1 = 0.9, double beta2 = 0.999,
                          double epsilon = 1e-8);
        
            void update(Layer& layer) override;
        
        private:
            struct State {
                long long t = 0; // Updates seen by this layer so far.
                Eigen::MatrixXd m_w, v_w;   // Weight moments.
                Eigen::VectorXd m_b, v_b;   // Bias moments.
            };
        
            std::unordered_map<Layer *, State> state_;
        
            double beta1_;
            double beta2_;
            double epsilon_;
    };
} // namespace nn
