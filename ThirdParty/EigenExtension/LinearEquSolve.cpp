#include "LinearEquSolve.h"
#include <Eigen/Eigen>
#define SZH_DEBUG_FLAG 1

#if SZH_DEBUG_FLAG
#include <iostream>
#endif

// solve the equation Ax = b by conjugate_gradient
void eigen_solve_conjugate_gradient(
    double * a,
    double * b,
    double * x,
    size_t ndim
)
{
    // first, we should map the input data (avoid data copy)
    // then, solve Ax = b by eigen.
    // finally, export the result.
    Eigen::Map<Eigen::MatrixXd> mat_a(a, ndim, ndim);
    Eigen::Map<Eigen::VectorXd> vec_b(b, ndim);
    Eigen::Map<Eigen::VectorXd> vec_x(x, ndim);
    Eigen::VectorXd init_gauss = 1e-2 * Eigen::VectorXd::Random(ndim);
    Eigen::ConjugateGradient<Eigen::MatrixXd, Eigen::Lower|Eigen::Upper> cg;
    cg.setMaxIterations(20 * ndim);
    cg.compute(mat_a);
    vec_x = cg.solveWithGuess(vec_b, init_gauss);
}

void eigen_solve_llt(
    double * a,
    double * b,
    double * x,
    size_t ndim
)
{
    Eigen::LLT<Eigen::MatrixXd> llt;
    Eigen::Map<Eigen::MatrixXd> mat_a(a, ndim, ndim);
    Eigen::Map<Eigen::VectorXd> vec_b(b, ndim);
    Eigen::Map<Eigen::VectorXd> vec_x(b, ndim);

    std::cout << "Here is the matrix A:\n" << mat_a << std::endl;
    std::cout << "Here is the right hand side b:\n" << vec_b << std::endl;
    std::cout << "Computing LLT decomposition..." << std::endl;
    llt.compute(mat_a);
    vec_x = llt.solve(vec_b);
    std::cout << "The solution is:\n" << vec_x << std::endl;
}

void eigen_solve_colPivHouseholderQr(
    double * a,
    double * b,
    double * x,
    size_t ndim
)
{
    Eigen::Map<Eigen::MatrixXd> mat_a(a, ndim, ndim);
    Eigen::Map<Eigen::VectorXd> vec_b(b, ndim);
    Eigen::Map<Eigen::VectorXd> vec_x(b, ndim);
    vec_x = mat_a.colPivHouseholderQr().solve(vec_b);
}

void eigen_solve_fullPivHouseholderQr(
    double * a,
    double * b,
    double * x,
    size_t ndim
)
{
    Eigen::Map<Eigen::MatrixXd> mat_a(a, ndim, ndim);
    Eigen::Map<Eigen::VectorXd> vec_b(b, ndim);
    Eigen::Map<Eigen::VectorXd> vec_x(b, ndim);
    vec_x = mat_a.fullPivHouseholderQr().solve(vec_b);
}