#pragma once
#include <cstring>

void eigen_solve_conjugate_gradient(
    double * a,
    double * b,
    double * x,
    size_t ndim
);

void eigen_solve_llt(
    double * a,
    double * b,
    double * x,
    size_t ndim
);

void eigen_solve_colPivHouseholderQr(
    double * a,
    double * b,
    double * x,
    size_t ndim
);