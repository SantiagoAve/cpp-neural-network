#include <iostream>
#include <eigen/Dense>

int main() {

    // Con esto se crea la matriz A.
    Eigen::Matrix2d matA;
    matA << 1, 2,
            3, 4;

    // Con esto se crea la matriz B.
    Eigen::Matrix2d matB;
    matB << 2, 0,
            1, 2;

    // Así se multiplica.
    Eigen::Matrix2d result = matA * matB;

    std::cout << "El resultado de la multiplicación es:\n" << result << std::endl;

    return 0;
}