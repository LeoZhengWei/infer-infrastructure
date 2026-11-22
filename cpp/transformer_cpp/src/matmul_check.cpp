#include <Eigen/Dense>
#include <iostream>

void matmul_check() {
    Eigen::MatrixXf A(2, 2);
    A << 1, 2, 3, 4;
    Eigen::MatrixXf B(2, 2);
    B << 5, 6, 7, 8;
    Eigen::MatrixXf C = A * B;
    std::cout << "C = " << C << std::endl;
}

int main() {
    matmul_check();
    return 0;
}