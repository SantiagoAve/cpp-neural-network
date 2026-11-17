/*
    ACTIVATION & LOSS TESTS:
    Value checks against known results plus derivative checks against central finite
    differences. Exit code 0 = all checks passed.
*/
#include "Activation.hpp"
#include "Loss.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>

namespace {
    int failures = 0;

    void check(bool condition, const std::string & what) {
        if (!condition) {
            ++failures;
            std::cerr << "FAIL: " << what << '\n';
        }
    }

    void check_close(double a, double b, double tol, const std::string & what) {
        const double scale = std::max(1.0, std::max(std::abs(a), std::abs(b)));
        check(std::abs(a - b) <= tol * scale, what + " (a=" + std::to_string(a) 
                                 + ", b=" + std::to_string(b) + ")");
    }

    // Central finite difference of a scalar function of one double.
    double numeric_derivative(const std::function<double(double)> & f, double x, double eps = 1e-6) {
        return (f(x + eps) - f(x - eps)) / (2.0 * eps);
    }
} // (Anonymous namespace)

int main() {
    using namespace nn;

    // Deterministic test grid, from deep negative to deep positive.
    Eigen::MatrixXd z(1, 9);
    z << -4.0, -2.5, -1.0, -0.25, 0.0, 0.25, 1.0, 2.5, 4.0;

    // --- Sigmoid ---
    const Eigen::MatrixXd sig = Activation::sigmoid().function(z);
    check_close(sig(0, 4), 0.5, 1e-12, "sigmoid(0) = 0.5");
    // Sigmoid(-4) ~ 0.018 and sigmoid(4) ~ 0.982: close to saturation.
    check(sig(0, 0) < 0.02 && sig(0, 8) > 0.98, "sigmoid saturates at +-4");
    for (int i = 0; i < z.size(); ++i) {
        const double x = z(0, i);
        const double numeric = numeric_derivative([&x](double t) {
            Eigen::MatrixXd m(1, 1);
            m << t;
            return Activation::sigmoid().function(m)(0, 0);
        }, x);
        const double analytic = Activation::sigmoid().derivative(z)(0, i);
        check_close(analytic, numeric, 1e-5, "sigmoid' matches finite differences");
    }

    // --- ReLU ---
    const Eigen::MatrixXd rel = Activation::relu().function(z);
    for (int i = 0; i < z.size(); ++i) {
        const double expected = std::max(0.0, z(0, i));
        check_close(rel(0, i), expected, 1e-12, "relu(max(z, 0))");
        // Derivative: 0 for z <= 0, 1 for z > 0.
        const double expected_deriv = z(0, i) > 0.0 ? 1.0 : 0.0;
        check_close(Activation::relu().derivative(z)(0, i), expected_deriv, 1e-12,
                    "relu'(z)");
    }

    // --- Tanh ---
    const Eigen::MatrixXd th = Activation::tanh().function(z);
    check_close(th(0, 4), 0.0, 1e-12, "tanh(0) = 0");
    check_close(th(0, 6), std::tanh(1.0), 1e-12, "tanh(1) matches std::tanh");
    for (int i = 0; i < z.size(); ++i) {
        const double x = z(0, i);
        const double numeric = numeric_derivative([&x](double t) {
            Eigen::MatrixXd m(1, 1);
            m << t;
            return Activation::tanh().function(m)(0, 0);
        }, x);
        const double analytic = Activation::tanh().derivative(z)(0, i);
        check_close(analytic, numeric, 1e-5, "tanh' matches finite differences");
    }

    // --- Identity ---
    const Eigen::MatrixXd id = Activation::identity().function(z);
    check(id.isApprox(z), "identity(z) = z");
    check(Activation::identity().derivative(z).isApproxToConstant(1.0),
          "identity'(z) = 1");

    // --- MSE value -----------------------------------------------------------
    const Eigen::MatrixXd y{{1.0, 0.0}, {0.0, 2.0}};
    const Eigen::MatrixXd p{{0.0, 0.0}, {0.0, 0.0}};
    // Mean(((1-0) ^ 2, 0, 0, (2 - 0) ^ 2)) = (1 + 0 + 0 + 4) / 4 = 1.25
    check_close(Loss::mean_squared(y, p), 1.25, 1e-12, "MSE known value");

    // --- MSE derivative vs Finite differences ---
    // Perturb P element (0, 0) and verify the analytic gradient entry.
    Eigen::MatrixXd p2 = p;
    p2(0, 0) += 1e-6;
    const double numeric = (Loss::mean_squared(y, p2) - Loss::mean_squared(y, p)) / 1e-6;
    const double analytic = Loss::mean_squared_derivative(y, p)(0, 0);
    check_close(analytic, numeric, 1e-4, "MSE' matches finite differences");

    // --- Shape validation ---
    bool threw = false;
    try {
        (void)Loss::mean_squared(y, Eigen::MatrixXd::Zero(3, 2));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "MSE rejects mismatched shapes");

    if (failures == 0) {
        std::cout << "All activation/loss checks passed.\n";
        return EXIT_SUCCESS;
    }
    std::cerr << failures << " check(s) failed.\n";
    return EXIT_FAILURE;
}
