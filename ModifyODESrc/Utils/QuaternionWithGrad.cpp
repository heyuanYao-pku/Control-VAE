#include "QuaternionWithGrad.h"
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <cmath>
#include <random>
#include <cstring>
#include <algorithm>
#define MY_DEBUG_OUTPUT_FLAG 0
#if MY_DEBUG_OUTPUT_FLAG
#include <iostream>
#endif

void minus_vector(
    const double * x_in,
    double * x_out,
    size_t dim
)
{
    for (size_t i = 0; i < dim; i++)
    {
        x_out[i] = -x_in[i];
    }
}

void mat3_set_as_eye_single(
    double * x
)
{
    x[0] = 1; x[1] = 0; x[2] = 0;
    x[3] = 0; x[4] = 1; x[5] = 0;
    x[6] = 0; x[7] = 0; x[8] = 1;
}

void mat3_vec3_multiply_single(
    const double * a,
    const double * x,
    double * b
)
{
    for (int i = 0; i < 3; i++)
    {
        b[i] = 0;
        for(int j = 0; j < 3; j++)
        {
            b[i] += a[3 * i + j] * b[j];
        }
    }
}

void mat3_vec3_multiply_impl(
    const double * a,
    const double * x,
    double * b,
    size_t num_mat
)
{
    for (size_t i = 0; i < num_mat; i++)
    {
        mat3_vec3_multiply_single(a + 9 * i, x + 3 * i, b + 3 * i);
    }
}

void mat3_vec3_multiply_backward_single(
    const double * a,
    const double * x,
    const double * grad_in,
    double * grad_a,
    double * grad_x
)
{
    // b[0] = a[0] * x[0] + a[1] * x[1] + a[2] * x[2]
    // b[1] = a[3] * x[0] + a[4] * x[1] + a[5] * x[2]
    // b[2] = a[6] * x[0] + a[7] * x[2] + a[8] * x[2]
    grad_a[0] = grad_in[0] * x[0];
    grad_a[1] = grad_in[0] * x[1];
    grad_a[2] = grad_in[0] * x[2];
    grad_a[3] = grad_in[1] * x[0];
    grad_a[4] = grad_in[1] * x[1];
    grad_a[5] = grad_in[1] * x[2];
    grad_a[6] = grad_in[2] * x[0];
    grad_a[7] = grad_in[2] * x[1];
    grad_a[8] = grad_in[2] * x[2];

    grad_x[0] = grad_in[0] * a[0] + grad_in[1] * a[3] + grad_in[2] * a[6];
    grad_x[1] = grad_in[0] * a[1] + grad_in[1] * a[4] + grad_in[2] * a[7];
    grad_x[2] = grad_in[0] * a[2] + grad_in[1] * a[5] + grad_in[2] * a[8];
}

void mat3_vec3_multiply_backward(
    const double * a,
    const double * x,
    const double * grad_in,
    double * grad_a,
    double * grad_x,
    size_t num_mat
)
{
    for(size_t i = 0; i < num_mat; i++)
    {
        mat3_vec3_multiply_backward_single(
            a + 9 * i,
            x + 3 * i,
            grad_in + 3 * i,
            grad_a + 9 * i,
            grad_x + 3 * i
        );
    }
}

void quat_multiply_single(
    const double * q1,
    const double * q2,
    double * q
)
{
    double x1 = q1[0], y1 = q1[1], z1 = q1[2], w1 = q1[3];
    double x2 = q2[0], y2 = q2[1], z2 = q2[2], w2 = q2[3];
    q[0] = + w1 * x2 - z1 * y2 + y1 * z2 + x1 * w2;
    q[1] = + z1 * x2 + w1 * y2 - x1 * z2 + y1 * w2;
    q[2] = - y1 * x2 + x1 * y2 + w1 * z2 + z1 * w2;
    q[3] = - x1 * x2 - y1 * y2 - z1 * z2 + w1 * w2;

    // here we can flip the quaternion by w..
    #if FLIP_QUAT_AT_MULTIPLY
    if (q[3] < 0)
    {
        q[0] = -q[0];
        q[1] = -q[1];
        q[2] = -q[2];
        q[3] = -q[3];
    }
    #endif
}

void quat_multiply_forward(
    const double * q1,
    const double * q2,
    double * q,
    size_t num_quat
)
{
    for(size_t i=0; i<num_quat; i++)
    {
        quat_multiply_single(q1 + 4 * i, q2 + 4 * i, q + 4 * i);
    }
}

void quat_multiply_backward_single(
    const double * q1,
    const double * q2,
    const double * grad_q, // \frac{\partial L}{\partial q_x, q_y, q_z, q_w}
    double * grad_q1,
    double * grad_q2
)
{
    double x1 = q1[0], y1 = q1[1], z1 = q1[2], w1 = q1[3];
    double x2 = q2[0], y2 = q2[1], z2 = q2[2], w2 = q2[3];
    #if FLIP_QUAT_AT_MULTIPLY
        double w = - x1 * x2 - y1 * y2 - z1 * z2 + w1 * w2;
        double flag = 1.0;
        if (w < 0)
        {
            flag = -1.0;
        }
        double gx = flag * grad_q[0], gy = flag * grad_q[1], gz = flag * grad_q[2], gw = flag * grad_q[3];
    #else
        double gx = grad_q[0], gy = grad_q[1], gz = grad_q[2], gw = grad_q[3];
    #endif

    grad_q1[0] = + gx * w2 - gy * z2 + gz * y2 - gw * x2;
    grad_q1[1] = + gx * z2 + gy * w2 - gz * x2 - gw * y2;
    grad_q1[2] = - gx * y2 + gy * x2 + gz * w2 - gw * z2;
    grad_q1[3] = + gx * x2 + gy * y2 + gz * z2 + gw * w2;

    grad_q2[0] = + gx * w1 + gy * z1 - gz * y1 - gw * x1;
    grad_q2[1] = - gx * z1 + gy * w1 + gz * x1 - gw * y1;
    grad_q2[2] = + gx * y1 - gy * x1 + gz * w1 - gw * z1;
    grad_q2[3] = + gx * x1 + gy * y1 + gz * z1 + gw * w1;
}

void quat_multiply_backward(
    const double * q1,
    const double * q2,
    const double * grad_q,
    double * grad_q1,
    double * grad_q2,
    size_t num_quat
)
{
    for(size_t i=0; i<num_quat; i++)
    {
        quat_multiply_backward_single(
            q1 + 4 * i, q2 + 4 * i, grad_q + 4 * i, grad_q1 + 4 * i, grad_q2 + 4 * i);
    }
}


void quat_apply_single(
    const double * q,
    const double * v,
    double * o
)
{
    /*
    precompute in python sympy
    import sympy
    qx, qy, qz, qw, vx, vy, vz = sympy.symbols("qx qy qz qw vx vy vz")
    u = sympy.Matrix([qx, qy, qz])
    v = sympy.Matrix([vx, vy, vz])
    t = 2 * u.cross(v)
    o = v + qw * t + u.cross(t)

    print(o)

    Matrix([[qw*(2*qy*vz - 2*qz*vy) + qy*(2*qx*vy - 2*qy*vx) - qz*(-2*qx*vz + 2*qz*vx) + vx],
    [qw*(-2*qx*vz + 2*qz*vx) - qx*(2*qx*vy - 2*qy*vx) + qz*(2*qy*vz - 2*qz*vy) + vy],
    [qw*(2*qx*vy - 2*qy*vx) + qx*(-2*qx*vz + 2*qz*vx) - qy*(2*qy*vz - 2*qz*vy) + vz]])
    */
    double qx = q[0], qy = q[1], qz = q[2], qw = q[3];
    double vx = v[0], vy = v[1], vz = v[2];
    o[0] = qw*(2*qy*vz - 2*qz*vy) + qy*(2*qx*vy - 2*qy*vx) - qz*(-2*qx*vz + 2*qz*vx) + vx;
    o[1] = qw*(-2*qx*vz + 2*qz*vx) - qx*(2*qx*vy - 2*qy*vx) + qz*(2*qy*vz - 2*qz*vy) + vy;
    o[2] = qw*(2*qx*vy - 2*qy*vx) + qx*(-2*qx*vz + 2*qz*vx) - qy*(2*qy*vz - 2*qz*vy) + vz;
}

void quat_apply_forward(
    const double * q,
    const double * v,
    double * o,
    size_t num_quat
)
{
    for(size_t i = 0; i < num_quat; i++)
    {
        quat_apply_single(q + i * 4, v + i * 3, o + i * 3);
    }
}

// Add by Yulong Zhang
void quat_apply_forward_one2many(
    const double * q,
    const double * v,
    double * o,
    size_t num_quat
)
{
    for(size_t i = 0; i < num_quat; i++)
    {
        quat_apply_single(q, v + i * 3, o + i * 3);
    }
}

void quat_apply_backward_single(
    const double * q,
    const double * v,
    const double * o_grad,
    double * q_grad,
    double * v_grad
)
{
    /*
    pre-compute in python sympy.
    print(o.diff(qx))
    print(o.diff(qy))
    print(o.diff(qz))
    print(o.diff(qw))
    print(o.diff(vx))
    print(o.diff(vy))
    print(o.diff(vz))

    o.diff(qx) = Matrix(((2*qy*vy + 2*qz*vz), (-2*qw*vz - 4*qx*vy + 2*qy*vx), (2*qw*vy - 4*qx*vz + 2*qz*vx)))
    o.diff(qy) = Matrix(((2*qw*vz + 2*qx*vy - 4*qy*vx), (2*qx*vx + 2*qz*vz), (-2*qw*vx - 4*qy*vz + 2*qz*vy)))
    o.diff(qz) = Matrix(((-2*qw*vy + 2*qx*vz - 4*qz*vx), (2*qw*vx + 2*qy*vz - 4*qz*vy), (2*qx*vx + 2*qy*vy)))
    o.diff(qw) = Matrix(((2*qy*vz - 2*qz*vy), (-2*qx*vz + 2*qz*vx), (2*qx*vy - 2*qy*vx)))
    o.diff(vx) = Matrix(((-2*qy**2 - 2*qz**2 + 1), (2*qw*qz + 2*qx*qy), (-2*qw*qy + 2*qx*qz)))
    o.diff(vy) = Matrix(((-2*qw*qz + 2*qx*qy), (-2*qx**2 - 2*qz**2 + 1), (2*qw*qx + 2*qy*qz)))
    o.diff(vz) = Matrix(((2*qw*qy + 2*qx*qz), (-2*qw*qx + 2*qy*qz), (-2*qx**2 - 2*qy**2 + 1)))
    */

    double qx = q[0], qy = q[1], qz = q[2], qw = q[3];
    double vx = v[0], vy = v[1], vz = v[2];
    q_grad[0] = o_grad[0] * (2*qy*vy + 2*qz*vz)              + o_grad[1] * (-2*qw*vz - 4*qx*vy + 2*qy*vx)    + o_grad[2] * (2*qw*vy - 4*qx*vz + 2*qz*vx);
    q_grad[1] = o_grad[0] * (2*qw*vz + 2*qx*vy - 4*qy*vx)    + o_grad[1] * (2*qx*vx + 2*qz*vz)               + o_grad[2] * (-2*qw*vx - 4*qy*vz + 2*qz*vy);
    q_grad[2] = o_grad[0] * (-2*qw*vy + 2*qx*vz - 4*qz*vx)   + o_grad[1] * (2*qw*vx + 2*qy*vz - 4*qz*vy)     + o_grad[2] * (2*qx*vx + 2*qy*vy);
    q_grad[3] = o_grad[0] * (2*qy*vz - 2*qz*vy)              + o_grad[1] * (-2*qx*vz + 2*qz*vx)              + o_grad[2] * (2*qx*vy - 2*qy*vx);
    v_grad[0] = o_grad[0] * (-2*qy*qy - 2*qz*qz + 1)         + o_grad[1] * (2*qw*qz + 2*qx*qy)               + o_grad[2] * (-2*qw*qy + 2*qx*qz);
    v_grad[1] = o_grad[0] * (-2*qw*qz + 2*qx*qy)             + o_grad[1] * (-2*qx*qx - 2*qz*qz + 1)          + o_grad[2] * (2*qw*qx + 2*qy*qz);
    v_grad[2] = o_grad[0] * (2*qw*qy + 2*qx*qz)              + o_grad[1] * (-2*qw*qx + 2*qy*qz)              + o_grad[2] * (-2*qx*qx - 2*qy*qy + 1);
}

void quat_apply_backward(
    const double * q,
    const double * v,
    const double * o_grad,
    double * q_grad,
    double * v_grad,
    size_t num_quat
)
{
    for(size_t i = 0; i < num_quat; i++)
    {
        quat_apply_backward_single(q + i * 4, v + i * 3, o_grad + i * 3, q_grad + i * 4, v_grad + i * 3);
    }
}

void flip_quat_by_w_forward_impl(
    const double * q,
    double * q_out,
    size_t num_quat
)
{
    for(size_t i = 0; i < num_quat; i++)
    {
        double flag = 1.0;
        if (q[i * 4 + 3] < 0)
        {
            flag = -1.0;
        }
        for(size_t j = 0; j < 4; j++)
        {
            q_out[i * 4 + j] = flag * q[i * 4 + j];
        }
    }
}

void flip_quat_by_w_backward_impl(
    const double * q,
    const double * grad_in,
    double * grad_out,
    size_t num_quat
)
{
    for(size_t i = 0; i < num_quat; i++)
    {
        double flag = 1.0;
        if (q[i * 4 + 3] < 0)
        {
            flag = -1.0;
        }
        for(size_t j = 0; j < 4; j++)
        {
            grad_out[i * 4 + j] = flag * grad_in[i * 4 + j];
        }
    }
}

void quat_to_vec6d_single(
    const double * q,
    double * vec6d
)
{
    double x = q[0], y = q[1], z = q[2], w = q[3];
    double x2 = x * x, y2 = y * y, z2 = z * z, w2 = w * w;
    double xy = x * y, zw = z * w, xz = x * z, yw = y * w, yz = y * z, xw = x * w;

    vec6d[0] = x2 - y2 - z2 + w2;  vec6d[1] = 2 * (xy - zw);
    vec6d[2] = 2 * (xy + zw);      vec6d[3] = - x2 + y2 - z2 + w2;
    vec6d[4] = 2 * (xz - yw);      vec6d[5] = 2 * (yz + xw);
}

void quat_to_vec6d_impl(const double * q, double * vec6d, size_t num_quat)
{
    for(size_t i = 0; i < num_quat; i++)
    {
        quat_to_vec6d_single(q + 4 * i, vec6d + 6 * i);
    }
}

void quat_to_matrix_forward_single(
    const double * q,
    double * mat
)
{
    double x = q[0], y = q[1], z = q[2], w = q[3];
    double x2 = x * x, y2 = y * y, z2 = z * z, w2 = w * w;
    double xy = x * y, zw = z * w, xz = x * z, yw = y * w, yz = y * z, xw = x * w;

    mat[0] = x2 - y2 - z2 + w2;  mat[1] = 2 * (xy - zw);        mat[2] = 2 * (xz + yw);
    mat[3] = 2 * (xy + zw);      mat[4] = - x2 + y2 - z2 + w2;  mat[5] = 2 * (yz - xw);
    mat[6] = 2 * (xz - yw);      mat[7] = 2 * (yz + xw);        mat[8] = - x2 - y2 + z2 + w2;
}

void quat_to_matrix_impl(
    const double * q,
    double * mat,
    size_t num_quat
)
{
    for(size_t i = 0; i < num_quat; i++)
    {
        quat_to_matrix_forward_single(q + 4 * i, mat + 9 * i);
    }
}

// Add by Yulong Zhang
void six_dim_mat_to_quat_single(
    const double * mat,
    double * quat
)
{
    double x1 = mat[0], x2 = mat[1], y1 = mat[2], y2 = mat[3], z1 = mat[4], z2 = mat[5];
    double x3 = y1*z2-y2*z1, y3 = -x1*z2+x2*z1, z3 = x1*y2-x2*y1;
    x2 = y3*z1-y1*z3; y2=-x3*z1+x1*z3; z2=x3*y1-x1*y3;
    double norm1 = std::sqrt(x1*x1+y1*y1+z1*z1), norm2 = std::sqrt(x2*x2+y2*y2+z2*z2), norm3=std::sqrt(x3*x3+y3*y3+z3*z3);
    x1 /= norm1; x2 /= norm2; x3 /= norm3;
    y1 /= norm1; y2 /= norm2; y3 /= norm3;
    z1 /= norm1; z2 /= norm2; z3 /= norm3;
    double q[4], w[4];
    w[0] = +x1+y2+z3;
    w[1] = +x1-y2-z3;
    w[2] = -x1+y2-z3;
    w[3] = -x1-y2+z3;
    size_t idx = 0;
    double val = x1;
    for(size_t i=1;i<4;i++){
        if (w[i] > val){
            val = w[i];
            idx = i;
        }
    }
    if(idx==0){
        q[0] = std::sqrt((1+w[0])/4);
        q[1] = (z2-y3)/(4*q[0]);
        q[2] = (x3-z1)/(4*q[0]);
        q[3] = (y1-x2)/(4*q[0]);
    }
    else if(idx==1){
        q[1] = std::sqrt((1+w[1])/4);
        q[0] = (z2-y3)/(4*q[1]);
        q[2] = (x2+y1)/(4*q[1]);
        q[3] = (x3+z1)/(4*q[1]);
    }
    else if(idx==2){
        q[2] = std::sqrt((1+w[2])/4);
        q[0] = (x3-z1)/(4*q[2]);
        q[1] = (x2+y1)/(4*q[2]);
        q[3] = (y3+z2)/(4*q[2]);
    }
    else{
        q[3] = std::sqrt((1+w[3])/4);
        q[0] = (y1-x2)/(4*q[3]);
        q[1] = (x3+z1)/(4*q[3]);
        q[2] = (y3+z2)/(4*q[3]);
    }
    quat[0] = q[1];
    quat[1] = q[2];
    quat[2] = q[3];
    quat[3] = q[0];
}

void six_dim_mat_to_quat_impl(
    const double * mat,
    double * q,
    size_t num_quat
)
{
    for (size_t i = 0; i < num_quat; i++)
    {
        six_dim_mat_to_quat_single(mat + 6 * i, q + 4 * i);
    }
}

void quat_to_matrix_backward_single(
    const double * q,
    const double * grad_in,
    double * grad_out
)
{
    /*
    Matrix(2*x, 2*y, 2*z, 2*y, -2*x, -2*w, 2*z, 2*w, -2*x)
    Matrix(-2*y, 2*x, 2*w, 2*x, 2*y, 2*z, -2*w, 2*z, -2*y)
    Matrix(-2*z, -2*w, 2*x, 2*w, -2*z, 2*y, 2*x, 2*y, 2*z)
    Matrix(2*w, -2*z, 2*y, 2*z, 2*w, -2*x, -2*y, 2*x, 2*w)
    */
    double x = q[0], y = q[1], z = q[2], w = q[3];
    double partial[4][9] = {
        {2*x, 2*y, 2*z, 2*y, -2*x, -2*w, 2*z, 2*w, -2*x},
        {-2*y, 2*x, 2*w, 2*x, 2*y, 2*z, -2*w, 2*z, -2*y},
        {-2*z, -2*w, 2*x, 2*w, -2*z, 2*y, 2*x, 2*y, 2*z},
        {2*w, -2*z, 2*y, 2*z, 2*w, -2*x, -2*y, 2*x, 2*w}
    };
    for(int i=0; i<4; i++)
    {
        double tmp = 0.0;
        for(int j=0; j<9; j++)
        {
            tmp += grad_in[j] * partial[i][j];
        }
        grad_out[i] = tmp;
    }
}

void quat_to_matrix_backward(
    const double * q,
    const double * grad_in,
    double * grad_out,
    size_t num_quat
)
{
    for(size_t i = 0; i < num_quat; i++)
    {
        quat_to_matrix_backward_single(q + 4 * i, grad_in + 9 * i, grad_out + 4 * i);
    }
}

void vector_cross_forward_single(
    const double * vect_A,
    const double * vect_B,
    double * cross_P
)
{
    cross_P[0] = vect_A[1] * vect_B[2] - vect_A[2] * vect_B[1];
    cross_P[1] = vect_A[2] * vect_B[0] - vect_A[0] * vect_B[2];
    cross_P[2] = vect_A[0] * vect_B[1] - vect_A[1] * vect_B[0];
}

void vector_cross_backward_single(
    const double * a,
    const double * b,
    const double * grad_in,
    double * grad_a,
    double * grad_b
)
{

}

void vector_cross_forward_impl(
    const double * a,
    const double * b,
    double * result,
    size_t num_vec
)
{
    for(size_t i = 0; i < num_vec; i++)
    {
        vector_cross_forward_single(a + 3 * i, b + 3 * i, result + 3 * i);
    }
}

void vector_cross_backward_impl(
    const double * a,
    const double * b,
    const double * grad_in,
    double * grad_a,
    double * grad_b,
    size_t num_vec
)
{
    for(size_t i = 0; i < num_vec; i++)
    {
        vector_cross_backward_single(
            a + 3 * i,
            b + 3 * i,
            grad_in + 3 * i,
            grad_a + 3 * i,
            grad_b + 3 * i
        );
    }
}

void vector_to_cross_matrix_single(
    const double * vec,
    double * mat
)
{
    double x0 = vec[0], x1 = vec[1], x2 = vec[2];
    mat[0] = 0;   mat[1] = -x2; mat[2] = x1;
    mat[3] = x2;  mat[4] = 0;   mat[5] = -x0;
    mat[6] = -x1; mat[7] = x0;  mat[8] = 0;
}

void vector_to_cross_matrix_impl(
    const double * vec,
    double * mat,
    size_t num_vec
)
{
    for(size_t i = 0; i < num_vec; i++)
    {
        vector_to_cross_matrix_single(vec + 3 * i, mat + 9 * i);
    }
}

void vector_to_cross_matrix_backward_single(
    const double * vec,
    const double * grad_in,
    double * grad_out
)
{
    grad_out[0] = -grad_in[5] + grad_in[7];
    grad_out[1] = grad_in[2] - grad_in[6];
    grad_out[2] = -grad_in[1] + grad_in[3];
}

void vector_to_cross_matrix_backward(
    const double * vec,
    const double * grad_in,
    double * grad_out,
    size_t num_vec
)
{
    for(size_t i = 0; i < num_vec; i++)
    {
        vector_to_cross_matrix_backward_single(vec + 3 * i, grad_in + 9 * i, grad_out + 3 * i);
    }
}

void quat_to_rotvec_single(
    const double * q,
    double & angle,
    double * rotvec
)
{
    // first, flip the quaternion by w.
    double ratio = 1.0;
    if (q[3] < 0)
    {
        ratio = -1.0;
    }
    double qx = ratio * q[0], qy = ratio * q[1], qz = ratio * q[2], qw = ratio * q[3];
    double ulen = std::sqrt(qx * qx + qy * qy + qz * qz);
    angle = 2.0 * std::atan2(ulen, qw);
    double scale = 1.0;
    if (std::abs(angle) < 1e-3)
    {
        double angle_2 = angle * angle;
        double angle_4 = angle_2 * angle_2;
        scale = 2 + angle_2 / 12 + 7 * angle_4 / 2880;
    }
    else
    {
        scale = angle / std::sin(0.5 * angle);
    }
    rotvec[0] = scale * qx;
    rotvec[1] = scale * qy;
    rotvec[2] = scale * qz;
}

void quat_to_rotvec_impl(
    const double * q,
    double * angle,
    double * rotvec,
    size_t num_quat
)
{
    for(size_t i = 0; i < num_quat; i++)
    {
        quat_to_rotvec_single(q + 4 * i, angle[i], rotvec + 3 * i);
    }
}

void quat_to_rotvec_backward_single(
    const double * q,
    double angle,
    const double * grad_in,
    double * grad_out
)
{
    double ratio = 1.0;
    if (q[3] < 0)
    {
        ratio = -1.0;
    }
    double x = ratio * q[0], y = ratio * q[1], z = ratio * q[2], w = ratio * q[3];
    double ulen = std::sqrt(x * x + y * y + z * z);
    double atan_val = 0.5 * angle;
    double atan_val2 = atan_val * atan_val;
    double atan_val3 = atan_val2 * atan_val;
    double atan_val4 = atan_val3 * atan_val;
    double ulen2 = ulen * ulen;
    double ulen3 = ulen2 * ulen;
    if (std::abs(angle) < 1e-3) // This branch checks OK.
    {
        double basic = 7*atan_val4/180 + atan_val2/3 + 2;
        double basic0 = 0.0;
        if (ulen > 1e-10) // avoid divide by zero..Note, when ulen is close to 0, we should use equivalent infinitesimal
        {
            basic0 = (7*atan_val3/15 + 2*atan_val)/(ulen*3);
        }
        else
        {
            basic0 = 2.0 / 3.0;
        }
        double basic1 = w*basic0;

        #if MY_DEBUG_OUTPUT_FLAG
        std::cout << "ulen = " << ulen << ", basic = " << basic << ", basic0 = " << basic0 << ", basic1 = " << basic1 << std::endl;
        #endif
        // partial L / partial x = (partial L / partial ox) * (partial ox / partial x)
        double basic_xyzw[4][3] = {{
            x*x*basic1 + basic,
            y*x*basic1,
            z*x*basic1
        },
        {
            x*y*basic1,
            y*y*basic1 + basic,
            z*y*basic1
        },
        {
            x*z*basic1,
            y*z*basic1,
            z*z*basic1 + basic
        },
        {
            -x*basic0*ulen2,
            -y*basic0*ulen2,
            -z*basic0*ulen2
        }};

        for(int i=0; i<4; i++)
        {
            double tmp = 0;
            for(int j=0; j<3; j++)
            {
                tmp += grad_in[j] * basic_xyzw[i][j];
            }
            grad_out[i] = tmp;
        }
    }
    else
    {
        double basic1 = 2*atan_val/ulen;
        double basic = 2*w/ulen2 + basic1 - 2*atan_val/ulen3;
        double basic2 = w*basic1 - 2;
        double basic_xyzw[4][3] = {{
            x*x*basic + basic1,
            x*y*basic,
            x*z*basic
        },
        {
            y*x*basic,
            y*y*basic + basic1,
            y*z*basic
        },
        {
            z*x*basic,
            z*y*basic,
            z*z*basic + basic1
        },
        {
            x*basic2,
            y*basic2,
            z*basic2
        }};
        for(int i=0; i<4; i++)
        {
            double tmp = 0;
            for(int j=0; j<3; j++)
            {
                tmp += grad_in[j] * basic_xyzw[i][j];
            }
            grad_out[i] = tmp;
        }
    }
    if (q[3] < 0)
    {
        for(int i=0; i<4; i++)
        {
            grad_out[i] = -grad_out[i];
        }
    }
}

void quat_to_rotvec_backward(
    const double * q,
    const double * angle,
    const double * grad_in,
    double * grad_out,
    size_t num_quat
)
{
    for(size_t i = 0; i < num_quat; i++)
    {
        quat_to_rotvec_backward_single(q + 4 * i, angle[i], grad_in + 3 * i, grad_out + 4 * i);
    }
}

void quat_from_rotvec_single(const double * rotvec, double * q)
{
    // q: qx, qy, qz, qw
    double angle = std::sqrt(rotvec[0] * rotvec[0] + rotvec[1] * rotvec[1] + rotvec[2] * rotvec[2]);
    double half_angle = 0.5 * angle;
    double ratio = 0.0;
    if (angle < 1e-3)
    {
        double angle2 = angle * angle;
        double angle4 = angle2 * angle2;
        ratio = 0.5 - angle2 / 48 + angle4 / 3840;
    }
    else
    {
        ratio = std::sin(half_angle) / angle;
    }
    q[0] = ratio * rotvec[0];
    q[1] = ratio * rotvec[1];
    q[2] = ratio * rotvec[2];
    q[3] = std::cos(half_angle);
}

void quat_from_rotvec_impl(const double * rotvec, double * q, size_t num_quat)
{
    for(size_t i = 0; i < num_quat; i++)
    {
        quat_from_rotvec_single(rotvec + 3 * i, q + 4 * i);
    }
}

void quat_from_rotvec_backward_single(const double * rotvec, const double * grad_in, double * grad_out)
{
    double sqr_angle = rotvec[0] * rotvec[0] + rotvec[1] * rotvec[1] + rotvec[2] * rotvec[2];
    double angle = std::sqrt(sqr_angle);
    double half_angle = 0.5 * angle;
    double x = rotvec[0], y = rotvec[1], z = rotvec[2];
    double x2 = x * x, y2 = y * y, z2 = z * z;
    double sin_half = std::sin(half_angle);
    double cos_half = std::cos(half_angle);
    grad_out[0] = grad_out[1] = grad_out[2] = 0;
    if (angle < 1e-3)
    {
        double sin_div = 0.5;
        double ratio_w = -0.5 * sin_div;
        double angle4 = angle * angle;
        double basic0 = sqr_angle / 960 - 24;
        double basic1 = 0.5 - sqr_angle / 48 + angle4 / 3840;
        //x*y*(sqr_angle/960 - 24)
        //-sqr_angle/48 + x*(x*sqr_angle/960 - x/24) + angle4/3840 + 0.5
        double grad_table[4][3] = {
            {x * x * basic0 + basic1, x * y * basic0         , x * z * basic0         },
            {x * y * basic0         , y * y * basic0 + basic1, y * z * basic0         },
            {x * z * basic0         , y * z * basic0         , z * z * basic0 + basic1},
            {x * ratio_w            , y * ratio_w            , z * ratio_w            }
        };
        for(int rj = 0; rj < 3; rj++)
        {
            for(int qi = 0; qi < 4; qi++)
            {
                grad_out[rj] += grad_table[qi][rj] * grad_in[qi];
            }
        }
    }
    else
    {
        double angle_inv = 1.0 / angle;
        double sqr_angle_inv = angle_inv * angle_inv;
        double sin_div = sin_half * angle_inv, cos_div = cos_half * angle_inv;
        double ratio_base = 0.5 * sqr_angle_inv * cos_half - sqr_angle_inv * sin_div;
        double ratio_x = x * ratio_base;
        double ratio_y = y * ratio_base;
        double ratio_z = z * ratio_base;
        double ratio_w = -0.5 * sin_div;
        double grad_table[4][3] = {
            {x * ratio_x + sin_div, y * ratio_x, z * ratio_x}, //  partial qx / partial x, partial qx / partial y, partial qx / partial z
            {x * ratio_y, y * ratio_y + sin_div, z * ratio_y}, // partial qy / partial x, partial qy / partial y, partial qy / partial z
            {x * ratio_z, y * ratio_z, z * ratio_z + sin_div}, // partial qz / partial x, partial qz / partial y, partial qz / partial z
            {x * ratio_w, y * ratio_w, z * ratio_w} // partial qw / partial x, partial qw / partial y, partial qw / partial z
        };

        for(int rj = 0; rj < 3; rj++)
        {
            for(int qi = 0; qi < 4; qi++)
            {
                grad_out[rj] += grad_table[qi][rj] * grad_in[qi];
            }
        }
    }
}

void quat_from_rotvec_backward_impl(
    const double * rotvec,
    const double * grad_in,
    double * grad_out,
    size_t num_quat
)
{
    for(size_t i = 0; i < num_quat; i++)
    {
        quat_from_rotvec_backward_single(rotvec + 3 * i, grad_in + 4 * i, grad_out + 3 * i);
    }
}

void quat_from_matrix_single(
    const double * mat,
    double * q
)
{
    double x1 = mat[0], x2 = mat[1], x3 = mat[2];
    double y1 = mat[3], y2 = mat[4], y3 = mat[5];
    double z1 = mat[6], z2 = mat[7], z3 = mat[8];
    double w[4] = {+x1+y2+z3, +x1-y2-z3, -x1+y2-z3, -x1-y2+z3};
    int idx = 0;
    double val = x1, div_val = 0;
    for(int i = 1; i < 4; i++)
    {
        if (w[i] > val)
        {
            val = w[i];
            idx = i;
        }
    }
    switch(idx)
    {
        case 0:
        {
            q[3] = 0.5 * std::sqrt(1+w[0]);
            div_val = 0.25 / q[3];
            q[0] = (z2-y3) * div_val;
            q[1] = (x3-z1) * div_val;
            q[2] = (y1-x2) * div_val;
            break;
        }
        case 1:
        {
            q[0] = 0.5 * std::sqrt(1+w[1]);
            div_val = 0.25 / q[0];
            q[3] = (z2-y3) * div_val;
            q[1] = (x2+y1) * div_val;
            q[2] = (x3+z1) * div_val;
            break;
        }
        case 2:
        {
            q[1] = 0.5 * std::sqrt(1+w[2]);
            div_val = 0.25 / q[1];
            q[3] = (x3-z1) * div_val;
            q[0] = (x2+y1) * div_val;
            q[2] = (y3+z2) * div_val;
            break;
        }
        case 3:
            q[2] = 0.5 * std::sqrt(1+w[3]);
            div_val = 0.25 / q[2];
            q[3] = (y1-x2) * div_val;
            q[0] = (x3+z1) * div_val;
            q[1] = (y3+z2) * div_val;
            break;
        default:
            break;
    }
    // (w, x, y, z). 0->3, 1->0, 2->1, 3->2
}

void quat_from_matrix_impl(
    const double * mat,
    double * q,
    size_t num_quat
)
{
    for(size_t i = 0; i < num_quat; i++)
    {
        quat_from_matrix_single(mat + 9 * i, q + 4 * i);
    }
}

void quat_from_matrix_backward_single(
    const double * mat,
    const double * grad_in,
    double * grad_out
)
{

}

void quat_from_matrix_backward_impl(
    const double * mat,
    const double * grad_in,
    double * grad_out,
    size_t num_quat
)
{
    for(size_t i = 0; i < num_quat; i++)
    {
        quat_from_matrix_backward_single(mat + 9 * i, grad_in + 4 * i, grad_out + 9 * i);
    }
}

void quat_to_hinge_angle_single(
    const double * q,
    const double * axis,
    double & angle
)
{
    double ratio = 1.0;
    if (q[3] < 0)
    {
        ratio = -1.0;
    }
    double qx = ratio * q[0], qy = ratio * q[1], qz = ratio * q[2], qw = ratio * q[3];
    double ax = axis[0], ay = axis[1], az = axis[2];
    double cos_val = qw, sin_val = std::sqrt(qx * qx + qy * qy + qz * qz);
    double dot_val = ax * qx + ay * qy + az * qz;
    double res = 0.0;
    if (dot_val >= 0)
    {
        res = 2 * std::atan2(sin_val, cos_val);
    }
    else
    {
        res = 2 * std::atan2(sin_val, -cos_val);
    }
    if (res > M_PI)
    {
        res -= 2 * M_PI;
    }
    angle = -res;
}

void quat_to_hinge_angle_forward(
    const double * q,
    const double * axis,
    double * angle,
    size_t num_quat
)
{
    for(size_t i = 0; i < num_quat; i++)
    {
        quat_to_hinge_angle_single(q + 4 * i, axis + 3 * i, angle[i]);
    }
}

void quat_to_hinge_angle_backward_single(
    const double * q,
    const double * axis,
    double grad_in,
    double * grad_out
)
{
    double ratio = 1.0;
    if (q[3] < 0)
    {
        ratio = -1.0;
    }
    else
    {
        grad_in = -grad_in;
    }

    double qx = ratio * q[0], qy = ratio * q[1], qz = ratio * q[2], qw = ratio * q[3];
    double ax = axis[0], ay = axis[1], az = axis[2];
    double cos_val = qw, sin_val = std::sqrt(qx * qx + qy * qy + qz * qz);
    double dot_val = ax * qx + ay * qy + az * qz;
    double dot_sign = 1.0;
    if (dot_val < 0)
    {
        dot_sign = -1.0;
    }
    double basic = 2 * qw / sin_val;
    grad_out[0] = grad_in * dot_sign * qx * basic;
    grad_out[1] = grad_in * dot_sign * qy * basic;
    grad_out[2] = grad_in * dot_sign * qz * basic;
    grad_out[3] = grad_in * dot_sign * -2* sin_val;
}

void quat_to_hinge_angle_backward(
    const double * q,
    const double * axis,
    const double * grad_in,
    double * grad_out,
    size_t num_quat
)
{
    for(size_t i = 0; i < num_quat; i++)
    {
        quat_to_hinge_angle_backward_single(q + 4 * i, axis + 3 * i, grad_in[i], grad_out + 4 * i);
    }
}

void quat_inv_single(
    const double * q,
    double * out_q
)
{
    // w = -1 * q[..., -1:]
    // xyz = q[..., :3]
    // return torch.cat([xyz, w], dim=-1)
    out_q[0] = q[0];
    out_q[1] = q[1];
    out_q[2] = q[2];
    out_q[3] = -q[3];
}

void quat_inv_impl(
    const double * q,
    double * out_q,
    size_t num_quat
)
{
    for(size_t i = 0; i < num_quat; i++)
    {
        quat_inv_single(q + 4 * i, out_q + 4 * i);
    }
}

void quat_inv_backward_single(
    const double * q,
    const double * grad_in,
    double * grad_out
)
{
    grad_out[0] = grad_in[0];
    grad_out[1] = grad_in[1];
    grad_out[2] = grad_in[2];
    grad_out[3] = -grad_in[3];
}

void quat_inv_backward_impl(
    const double * q,
    const double * grad_in,
    double * grad_out,
    size_t num_quat
)
{
    for(size_t i = 0; i < num_quat; i++)
    {
        quat_inv_backward_single(q + 4 * i, grad_in + 4 * i, grad_out + 4 * i);
    }
}

// This function is called by diffode when compute hinge angle..
void parent_child_quat_to_hinge_angle_single_func(
    const double * quat0,
    const double * quat1,
    const double * init_rel_quat_inv,
    const double * axis,
    double & angle,
    double * quat0_inv,
    double * dq,
    double * dquat
)
{
    quat_inv_single(quat0, quat0_inv);
    quat_multiply_single(quat0_inv, quat1, dq);
    quat_multiply_single(dq, init_rel_quat_inv, dquat);
    quat_to_hinge_angle_single(dquat, axis, angle);
}

void parent_child_quat_to_hinge_angle_single(
    const double * quat0,
    const double * quat1,
    const double * init_rel_quat_inv,
    const double * axis,
    double & angle
)
{
    // quat0_inv: torch.Tensor = quat_inv(quat0)
    // dq: torch.Tensor = quat_multiply(quat0_inv, quat1)  # relative quaternion between two bodies
    // dquat: torch.Tensor = quat_multiply(dq, self.init_rel_quat_inv)
    double quat0_inv[4], dq[4], dquat[4];
    parent_child_quat_to_hinge_angle_single_func(quat0, quat1, init_rel_quat_inv, axis, angle, quat0_inv, dq, dquat);
}

void parent_child_quat_to_hinge_angle(
    const double * quat0,
    const double * quat1,
    const double * init_rel_quat_inv,
    const double * axis,
    double * angle,
    size_t num_quat
)
{
    for(size_t i = 0; i < num_quat; i++)
    {
        parent_child_quat_to_hinge_angle_single(quat0 + 4 * i, quat1 + 4 * i, init_rel_quat_inv + 4 * i, axis + 3 * i, angle[i]);
    }
}

void parent_child_quat_to_hinge_angle_backward_single(
    const double * quat0,
    const double * quat1,
    const double * init_rel_quat_inv,
    const double * axis,
    double grad_in,
    double * quat0_grad,
    double * quat1_grad
)
{
    //ah, we also needs to do forward path..
    // quat_inv_single(quat0, quat0_inv);
    // quat_multiply_single(quat0_inv, quat1, dq);
    // quat_multiply_single(dq, init_rel_quat_inv, dquat);
    // quat_to_hinge_angle_single(dquat, axis, angle);
    double quat0_inv[4], dq[4], dquat[4], angle;
    parent_child_quat_to_hinge_angle_single_func(quat0, quat1, init_rel_quat_inv, axis, angle, quat0_inv, dq, dquat);

    // first, compute partial L / partial dquat
    double dquat_grad[4], dq_grad[4], quat0_inv_grad[4], dummy[4];
    quat_to_hinge_angle_backward_single(dquat, axis, grad_in, dquat_grad);

    // then, compute partial dquat / dq. we need not to consider the gradient for init_rel_quat_inv
    quat_multiply_backward_single(dq, init_rel_quat_inv, dquat_grad, dq_grad, dummy);

    // then, compute (partial dq / partial quat0_inv) and (partial dq / partial quat1)
    quat_multiply_backward_single(quat0_inv, quat1, dq_grad, quat0_inv_grad, quat1_grad);

    // finally, compute partial quat0_inv / partial quat0
    quat_inv_backward_single(quat0, quat0_inv_grad, quat0_grad);
}

void parent_child_quat_to_hinge_angle_backward(
    const double * quat0,
    const double * quat1,
    const double * init_rel_quat_inv,
    const double * axis,
    const double * grad_in,
    double * quat0_grad,
    double * quat1_grad,
    size_t num_quat
)
{
    for(size_t i = 0; i < num_quat; i++)
    {
        parent_child_quat_to_hinge_angle_backward_single(
            quat0 + 4 * i, quat1 + 4 * i, init_rel_quat_inv + 4 * i,
            axis + 3 * i, grad_in[i], quat0_grad + 4 * i, quat1_grad + 4 * i);
    }
}

void vector_normalize_single(
    const double * x,
    size_t ndim,
    double * result
)
{
    double tot_len = 0.0;
    for(size_t i = 0; i < ndim; i++)
    {
        tot_len += x[i] * x[i];
    }
    if (tot_len < 1e-10)
    {
        for(size_t i = 0; i < ndim; i++)
        {
            result[i] = x[i];
        }
    }
    else
    {
        tot_len = 1.0 / std::sqrt(tot_len);
        for(size_t i = 0; i < ndim; i++)
        {
            result[i] = tot_len * x[i];
        }
    }
}

void _vector_normalize_backward_with_known_tot_len(
    const double * x,
    size_t ndim,
    const double * grad_in,
    double * grad_out,
    double tot_len
)
{
    tot_len = std::sqrt(tot_len);
    double tot_len2 = tot_len * tot_len;
    double tot_len3 = tot_len2 * tot_len;
    double tot_len_inv = 1.0 / tot_len;
    double tot_len3_inv = 1.0 / tot_len3;

    double sum_value = 0.0;
    for(size_t i = 0; i < ndim; i++)
    {
        sum_value += x[i] * grad_in[i];
    }
    sum_value = -tot_len3_inv * sum_value;
    for(size_t i = 0; i < ndim; i++)
    {
        grad_out[i] = grad_in[i] * tot_len_inv + sum_value * x[i];
    }
}

void vector_normalize_backward_single(
    const double * x,
    size_t ndim,
    const double * grad_in,
    double * grad_out
)
{
    double tot_len = 0.0;
    for(size_t i = 0; i < ndim; i++)
    {
        tot_len += x[i] * x[i];
    }
    if (tot_len < 1e-10)
    {
        for(size_t i = 0; i < ndim; i++)
        {
            grad_out[i] = grad_in[i];
        }
    }
    else
    {
        _vector_normalize_backward_with_known_tot_len(x, ndim, grad_in, grad_out, tot_len);
    }
}

void normalize_quaternion_impl(
    const double * q_in,
    double * q_out,
    size_t num_quat
)
{
    for(size_t i = 0; i < num_quat; i++)
    {
        double sum_value = 0.0;
        const double * q = q_in + 4 * i;
        double * o = q_out + 4 * i;
        for(int j = 0; j < 4; j++)
        {
            sum_value += q[j] * q[j];
        }
        sum_value = 1.0 / std::sqrt(sum_value);
        for(int j = 0; j < 4; j++)
        {
            o[j] = q[j] * sum_value;
        }
    }
}

void normalize_quaternion_backward_impl(
    const double * q_in,
    const double * grad_in,
    double * grad_out,
    size_t num_quat
)
{
    for(size_t i = 0; i < num_quat; i++)
    {
        vector_normalize_backward_single(q_in + 4 * i, 4, grad_in + 4 * i, grad_out + 4 * i);
    }
}

void quat_integrate_single(
    const double * q,
    const double * omega,
    double dt,
    double * result
)
{
    //omega = torch.cat([omega, torch.zeros(omega.shape[:-1] + (1,), dtype=omega.dtype, device=q.device)], -1)
    //delta_q = 0.5 * dt * quat_multiply(omega, q)
    //result = q + delta_q
    //result = quat_normalize(result)
    double omega_q[4] = {omega[0], omega[1], omega[2], 0.0}, delta_q[4], res_len = 0.0;
    quat_multiply_single(omega_q, q, delta_q);
    for(int i = 0; i < 4; i++)
    {
        result[i] = q[i] + 0.5 * dt * delta_q[i];
        res_len += result[i] * result[i];
    }
    res_len = 1.0 / std::sqrt(res_len);
    for(int i = 0; i < 4; i++)
    {
        result[i] = result[i] * res_len;
    }
}

void quat_integrate_impl(
    const double * q,
    const double * omega,
    double dt,
    double * result,
    size_t num_quat
)
{
    for(size_t i = 0; i < num_quat; i++)
    {
        quat_integrate_single(q + 4 * i, omega + 3 * i, dt, result + 4 * i);
    }
}

void quat_integrate_backward_single(
    const double * q,
    const double * omega,
    double dt,
    const double * grad_in,
    double * q_grad,
    double * omega_grad
)
{
    //omega = torch.cat([omega, torch.zeros(omega.shape[:-1] + (1,), dtype=omega.dtype, device=q.device)], -1)
    //delta_q = 0.5 * dt * quat_multiply(omega, q)
    //result = q + delta_q
    //result = quat_normalize(result)
    double omega_q[4] = {omega[0], omega[1], omega[2], 0.0}, delta_q[4], result[4];
    quat_multiply_single(omega_q, q, delta_q);
    for(int i = 0; i < 4; i++)
    {
        result[i] = q[i] + 0.5 * dt * delta_q[i];
    }

    // 1. compute normalize gradient, that is, (partial L / partial result) = (partial L / partial final_result) * (partial final_result / partial )
    double result_grad[4], delta_q_grad[4], delta_q_grad_ratio = 0.5 * dt;
    vector_normalize_backward_single(result, 4, grad_in, result_grad);

    // 2. compute add gradient  result[i] = q[i] + delta_q[i];
    for(int i = 0; i < 4; i++) q_grad[i] = result_grad[i];
    for(int i = 0; i < 4; i++) delta_q_grad[i] = delta_q_grad_ratio * result_grad[i];

    // 3. compute quaternion multiply gradient
    double omega_q_grad[4], q_tmp_grad[4];
    quat_multiply_backward_single(omega_q, q, delta_q_grad, omega_q_grad, q_tmp_grad);
    for(int i = 0; i < 4; i++) q_grad[i] += q_tmp_grad[i];

    // 4. compute omega gradient
    for(int i = 0; i < 3; i++) omega_grad[i] = omega_q_grad[i];
}

void quat_integrate_backward(
    const double * q,
    const double * omega,
    double dt,
    const double * grad_in,
    double * q_grad,
    double * omega_grad,
    size_t num_quat
)
{
    for(size_t i = 0; i < num_quat; i++)
    {
        quat_integrate_backward_single(q + 4 * i, omega + 3 * i, dt, grad_in + 4 * i, q_grad + 4 * i, omega_grad + 3 * i);
    }
}

// Add by Yulong Zhang
void calc_surface_distance_to_capsule_single(
    const double * relative_pos,
    const double radius,
    const double length,
    double & sd,
    double * normal
)
{
    double vec[3];
    vec[0] = relative_pos[0];
    vec[2] = relative_pos[2];
    if(std::abs(relative_pos[1]) < length/2.0)
    {
        vec[1] = 0.0;
    }
    else if(relative_pos[1] >= length/2.0)
    {
        vec[1] = relative_pos[1] - length/2.0;
    }
    else
    {
        vec[1] = relative_pos[1] + length/2.0;
    }
    sd = std::sqrt(vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]) - radius;
    vector_normalize_single(vec, 3, normal);
}

void calc_surface_distance_to_capsule(
    const double * relative_pos,
    size_t ndim,
    const double radius,
    const double length,
    double * sd,
    double * normal
)
{
    for(size_t i=0;i<ndim;i++)
    {
        calc_surface_distance_to_capsule_single(relative_pos + 3 * i, radius, length, sd[i], normal + 3 * i);
    }
}

void clip_vec_by_norm_forward_single(
    const double * x,
    double min_val,
    double max_val,
    double * result,
    size_t ndim
)
{
    for(size_t i = 0; i < ndim; i++)
    {
        if (x[i] < min_val) result[i] = min_val;
        else if (x[i] > max_val) result[i] = max_val;
        else result[i] = x[i];
    }
}

void clip_vec_by_length_forward(
    const double * x,
    double max_len,
    double * result,
    size_t ndim
)
{
    double tot_len = 0.0;
    for(size_t i = 0; i < ndim; i++)
    {
        tot_len += x[i] * x[i];
    }
    tot_len = std::sqrt(tot_len);
    if (tot_len <= max_len)
    {
        for(size_t i = 0; i < ndim; i++)
        {
            result[i] = x[i];
        }
    }
    else
    {
        // clip the result..
        tot_len = max_len / tot_len;
        for(size_t i = 0; i < ndim; i++)
        {
            result[i] = tot_len * x[i];
        }
    }
}

void clip_vec3_arr_by_length_forward(
    const double * x,
    const double * max_len,
    double * result,
    size_t num_vecs
)
{
    for(size_t i = 0; i < num_vecs; i++)
    {
        clip_vec_by_length_forward(x + 3 * i, max_len[i], result + 3 * i, 3);
    }
}

void clip_vec_by_length_backward(
    const double * x,
    double max_len,
    const double * grad_in,
    double * grad_out,
    size_t ndim
)
{
    double tot_len = 0.0;
    for(size_t i = 0; i < ndim; i++)
    {
        tot_len += x[i] * x[i];
    }
    if (tot_len <= max_len * max_len)
    {
        for(size_t i = 0; i < ndim; i++)
        {
            grad_out[i] = grad_in[i];
        }
    }
    else
    {
        _vector_normalize_backward_with_known_tot_len(x, ndim, grad_in, grad_out, tot_len);
        for(size_t i = 0; i < ndim; i++)
        {
            grad_out[i] *= max_len;
        }
    }
}

void clip_vec3_arr_by_length_backward(
    const double * x,
    const double * max_len,
    const double * grad_in,
    double * grad_out,
    size_t num_vecs
)
{
    for(size_t i = 0; i < num_vecs; i++)
    {
        clip_vec_by_length_backward(x + 3 * i, max_len[i], grad_in + 3 * i, grad_out + 3 * i, 3);
    }
}

void decompose_rotation_single(
    const double * q,
    const double * vb,
    double * result
)
{
    double raw_va[3] = {0, 0, 0}, va[3] = {0, 0, 0};
    double raw_rot_axis[3] = {0, 0, 0}, rot_axis[3] = {0, 0, 0}, axis_angle[3] = {0, 0, 0};
    quat_apply_single(q, vb, raw_va);
    vector_normalize_single(raw_va, 3, va);
    vector_cross_forward_single(va, vb, raw_rot_axis);
    vector_normalize_single(raw_rot_axis, 3, rot_axis);
    double angle = va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2];
    angle = std::clamp<double>(angle, -1.0, 1.0);
    angle = std::acos(angle);
    for(int i = 0; i < 3; i++) axis_angle[i] = angle * rot_axis[i];
    double tmp_quat[4];
    quat_from_rotvec_single(axis_angle, tmp_quat);
    quat_multiply_single(tmp_quat, q, result);
}

void decompose_rotation(
    const double * q,
    const double * v,
    double * result,
    size_t num_quat
)
{
    for(size_t i = 0; i < num_quat; i++)
    {
        decompose_rotation_single(q + 4 * i, v + 3 * i, result + 4 * i);
    }
}

void decompose_rotation_pair_single(
    const double * q,
    const double * vb,
    double * q_a,
    double * q_b
)
{
    // r_other = (Rotation(res, copy=False, normalize=False).inv() * Rotation(q, copy=False, normalize=False)).as_quat()
    decompose_rotation_single(
        q,
        vb,
        q_a
    );
    double a_inv[4];
    quat_inv_single(q_a, a_inv);
    quat_multiply_single(a_inv, q, q_b);
    for(int i = 0; i < 4; i++)
    {
        if (std::abs(q_a[i]) < 1e-14)
        {
            q_a[i] = 0;
        }
    }
    vector_normalize_single(q_a, 4, q_a);
    for(int i = 0; i < 4; i++)
    {
        if (std::abs(q_b[i]) < 1e-14)
        {
            q_b[i] = 0;
        }
    }
    vector_normalize_single(q_b, 4, q_b);
}

void decompose_rotation_pair(
    const double * q,
    const double * vb,
    double * q_a,
    double * q_b,
    size_t num_quat
)
{
    for(size_t i = 0; i < num_quat; i++)
    {
        decompose_rotation_pair_single(
            q + 4 * i,
            vb + 3 * i,
            q_a + 4 * i,
            q_b + 4 * i
        );
    }
}

void decompose_rotation_pair_one2many(
    const double * q,
    const double * vb,
    double * q_a,
    double * q_b,
    size_t num_quat
)
{
    for(size_t i = 0; i < num_quat; i++)
    {
        decompose_rotation_pair_single(
            q + 4 * i,
            vb,
            q_a + 4 * i,
            q_b + 4 * i
        );
    }
}

void decompose_rotation_backward_single(
    const double * q,
    const double * v,
    const double * grad_in,
    double * grad_q,
    double * grad_v
)
{

}

void decompose_rotation_backward(
    const double * q,
    const double * v,
    const double * grad_in,
    double * grad_q,
    double * grad_v,
    size_t num_quat
)
{
    for(size_t i = 0; i < num_quat; i++)
    {
        decompose_rotation_backward_single(
            q,
            v,
            grad_in,
            grad_q,
            grad_v
        );
    }
}

// ==================== for fast differentiable stable pd control =========================

static void _compute_parent_body_quaternion(
    const double * body_quat,
    double * parent_quat,
    const int * joint_to_parent_body,
    int num_joint
)
{
    for(int i = 0; i < num_joint; i++)
    {
        int parent_body = joint_to_parent_body[i];
        double * q = parent_quat + 4 * i;
        if (parent_body == -1)
        {
            q[0] = q[1] = q[2] = 0;
            q[3] = 1;
        }
        else
        {
            const double * qb = body_quat + 4 * parent_body;
            for(int j = 0; j < 4; j++) q[j] = qb[j];
        }
    }
}

static void _compute_child_body_quaternion(
    const double * body_quat,
    double * child_quat,
    const int * joint_to_child_body,
    int num_joint
)
{
    for(int i = 0; i < num_joint; i++)
    {
        int child_body = joint_to_child_body[i];
        double * q = child_quat + 4 * i;
        const double * qb = body_quat + 4 * child_body;
        for(int j = 0; j < 4; j++) q[j] = qb[j];
    }
}

static void _compute_parent_body_quaternion_backward(
    const double * grad_in,
    double * grad_out,
    const int * joint_to_parent_body,
    int num_joint
)
{
    for(int i = 0; i < num_joint; i++)
    {
        int parent_body = joint_to_parent_body[i];
        const double * q = grad_in + 4 * i;
        if (parent_body == -1) continue;
        double * qb = grad_out + 4 * parent_body;
        for(int j = 0; j < 4; j++) qb[j] += q[j];
    }
}

static void _compute_child_body_quaternion_backward(
    const double * grad_in,
    double * grad_out,
    const int * joint_to_child_body,
    int num_joint
)
{
    for(int i = 0; i < num_joint; i++)
    {
        int child_body = joint_to_child_body[i];
        const double * q = grad_in + 4 * i;
        double * qb = grad_out + 4 * child_body;
        for(int j = 0; j < 4; j++) qb[j] += q[j];
    }
}

static void _multiply_kp(
    const double * kp,
    const double * rotvec,
    double * torque,
    int num_joint
)
{
    for(int i = 0; i < num_joint; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            torque[3 * i + j] = kp[i] * rotvec[3 * i + j];
        }
    }
}

static void _multiply_kp_backward(
    const double * kp,
    const double * grad_in,
    double * grad_out,
    int num_joint
)
{
    _multiply_kp(kp, grad_in, grad_out, num_joint);
}

// Ah, as stable pd control will be called for many times, we need to allocate memory..

void StablePDControlMemoryNode::mem_alloc(size_t num_joint, size_t num_body)
{
    parent_quat = new double[num_joint * 4];
    child_quat = new double[num_joint * 4];
    parent_quat_inv = new double[num_joint * 4];
    local_quat = new double[num_joint * 4];
    local_quat_normalize = new double[num_joint * 4];

    inv_local_quat = new double[num_joint * 4];
    delta_quat = new double[num_joint * 4];
    rotvec = new double[num_joint * 3];
    angle = new double[num_joint];
    torque = new double[num_joint * 3];
    clip_torque = new double[num_joint * 3];
    global_joint_torque = new double[num_joint * 3];
    global_body_torque = new double[num_body * 3];
}

void StablePDControlMemoryNode::mem_dealloc()
{
    if (parent_quat != nullptr) {delete[] parent_quat; parent_quat = nullptr; }
    if (child_quat != nullptr) {delete[] child_quat; child_quat = nullptr; }
    if (parent_quat_inv != nullptr) {delete[] parent_quat_inv; parent_quat_inv = nullptr; }
    if (local_quat != nullptr) {delete[] local_quat; local_quat = nullptr; }
    if (local_quat_normalize != nullptr) {delete[] local_quat_normalize; local_quat_normalize = nullptr; }
    if (inv_local_quat != nullptr) {delete[] inv_local_quat; inv_local_quat = nullptr; }
    if (delta_quat != nullptr) {delete[] delta_quat; delta_quat = nullptr; }
    if (rotvec != nullptr) {delete[] rotvec; rotvec = nullptr; }
    if (angle != nullptr) {delete[] angle; angle = nullptr; }
    if (torque != nullptr) {delete[] torque; torque = nullptr; }
    if (clip_torque != nullptr) {delete[] clip_torque; clip_torque = nullptr; }
    if (global_joint_torque != nullptr) {delete[] global_joint_torque; global_joint_torque = nullptr; }
    if (global_body_torque != nullptr) {delete[] global_body_torque; global_body_torque = nullptr; }
}

void StablePDControlMemoryNode::zero(size_t num_joint, size_t num_body)
{
    memset(parent_quat, 0, sizeof(double) * num_joint * 4);
    memset(child_quat, 0, sizeof(double) * num_joint * 4);
    memset(parent_quat_inv, 0, sizeof(double) * num_joint * 4);
    memset(local_quat, 0, sizeof(double) * num_joint * 4);
    memset(local_quat_normalize, 0, sizeof(double) * num_joint * 4);

    memset(inv_local_quat, 0, sizeof(double) * num_joint * 4);
    memset(delta_quat, 0, sizeof(double) * num_joint * 4);
    memset(rotvec, 0, sizeof(double) * num_joint * 3);
    memset(angle, 0, sizeof(double) * num_joint);
    memset(torque, 0, sizeof(double) * num_joint * 3);
    memset(clip_torque, 0, sizeof(double) * num_joint * 3);
    memset(global_joint_torque, 0, sizeof(double) * num_joint * 3);
    memset(global_body_torque, 0, sizeof(double) * num_body * 3);
}

StablePDControlMemoryNode * StablePDControlMemoryNodeCreate(size_t num_joint, size_t num_body)
{
    StablePDControlMemoryNode * grad_mem_accum = new StablePDControlMemoryNode();
    grad_mem_accum->mem_alloc(num_joint, num_body);
    return grad_mem_accum;
}

void StablePDControlMemoryNodeDestroy(StablePDControlMemoryNode * node)
{
    if (node != nullptr)
    {
        node->mem_dealloc();
        delete node;
    }
}

// now, we should rewrite stable pd controller here, for higher performance
void stable_pd_control_forward(
    const double * body_quat,
    const double * target_quat,
    double * global_body_torque,
    TorqueAddHelper * helper,
    StablePDControlMemoryNode * _stable_pd_mem
)
{
    const int * joint_to_parent_body = helper->get_parent_body_c_ptr();
    const int * joint_to_child_body = helper->get_child_body_c_ptr();
    const double * kp = helper->get_kp_c_ptr();
    const double * max_len = helper->get_max_len_c_ptr();
    int num_joint = helper->GetJointCount();
    int num_body = helper->GetBodyCount();

    // 1. compute parent body quaternion.
    double * parent_quat = _stable_pd_mem->parent_quat;
    _compute_parent_body_quaternion(body_quat, parent_quat, joint_to_parent_body, num_joint);

    // 2. compute child body quaternion
    double * child_quat = _stable_pd_mem->child_quat;
    _compute_child_body_quaternion(body_quat, child_quat, joint_to_child_body, num_joint);

    // 3. compute inverse of parent quaternion
    double * parent_quat_inv = _stable_pd_mem->parent_quat_inv;
    quat_inv_impl(parent_quat, parent_quat_inv, num_joint);

    // 4. compute joint local quaternion, and normalize the local quaternion
    quat_multiply_forward(parent_quat_inv, child_quat, _stable_pd_mem->local_quat, num_joint);

    // 5. normalize the local quaternion
    normalize_quaternion_impl(_stable_pd_mem->local_quat, _stable_pd_mem->local_quat_normalize, num_joint);

    // 6. compute the inverse of local quaternion
    double * inv_local_quat = _stable_pd_mem->inv_local_quat;
    quat_inv_impl(_stable_pd_mem->local_quat_normalize, inv_local_quat, num_joint);

    // 7. compute difference between target pose and local quaternion
    double * delta_quat = _stable_pd_mem->delta_quat;
    quat_multiply_forward(target_quat, inv_local_quat, delta_quat, num_joint);

    // 8. convert the delta quat into axis angle format.
    double * rotvec = _stable_pd_mem->rotvec;
    double * angle = _stable_pd_mem->angle;
    quat_to_rotvec_impl(delta_quat, angle, rotvec, num_joint);

    // 9. multiply with kp
    double * torque = _stable_pd_mem->torque;
    _multiply_kp(kp, rotvec, torque, num_joint);

    // 10. clip the torque
    double * clip_torque = _stable_pd_mem->clip_torque;
    clip_vec3_arr_by_length_forward(torque, max_len, clip_torque, num_joint);
    // double * clip_torque = torque;

    // 11. rotate the torque to global coordinate
    double * global_joint_torque = _stable_pd_mem->global_joint_torque;
    quat_apply_forward(parent_quat, clip_torque, global_joint_torque, num_joint);

    // 12. add torque from joint to body.
    helper->add_torque_forward(global_body_torque, global_joint_torque);
}

void stable_pd_control_backward(
    const double * body_quat,
    const double * target_quat,
    const double * grad_in,
    double * grad_out_body_quat,
    double * grad_out_target_quat,
    TorqueAddHelper * helper,
    StablePDControlMemoryNode * _stable_pd_mem
)
{
    const int * joint_to_parent_body = helper->get_parent_body_c_ptr();
    const int * joint_to_child_body = helper->get_child_body_c_ptr();
    const double * kp = helper->get_kp_c_ptr();
    const double * max_len = helper->get_max_len_c_ptr();

    int num_joint = helper->GetJointCount();
    int num_body = helper->GetBodyCount();
    double * tmp_grad = new double[4 * num_body];

    // zero gradient
    StablePDControlMemoryNode * grad_mem_accum = StablePDControlMemoryNodeCreate(num_joint, num_body);
    grad_mem_accum->zero(num_joint, num_body);

    // 12. add_torque from joint to body: backward
    helper->backward(grad_in, grad_mem_accum->global_joint_torque);

    // 11. rotate the torque from local to global
    quat_apply_backward(_stable_pd_mem->parent_quat, _stable_pd_mem->clip_torque, grad_mem_accum->global_joint_torque,
        grad_mem_accum->parent_quat, grad_mem_accum->clip_torque, num_joint);

    // 10. clip the torque
    clip_vec3_arr_by_length_backward(_stable_pd_mem->torque, max_len, grad_mem_accum->clip_torque, grad_mem_accum->torque, num_joint);


    // 9. multiply with kp
    _multiply_kp_backward(kp, grad_mem_accum->torque, grad_mem_accum->rotvec, num_joint);

    // 8. convert the delta quat into axis angle format.
    quat_to_rotvec_backward(_stable_pd_mem->delta_quat, _stable_pd_mem->angle, grad_mem_accum->rotvec,
        grad_mem_accum->delta_quat, num_joint);


    // 7. compute difference between target pose and local quaternion
    quat_multiply_backward(target_quat, _stable_pd_mem->inv_local_quat,  grad_mem_accum->delta_quat,
        grad_out_target_quat, grad_mem_accum->inv_local_quat, num_joint);

    // 6. compute the inverse of local quaternion
    quat_inv_backward_impl(_stable_pd_mem->local_quat_normalize,
        grad_mem_accum->inv_local_quat, grad_mem_accum->local_quat_normalize, num_joint);

    // 5. normalize the local quaternion
    normalize_quaternion_backward_impl(_stable_pd_mem->local_quat, grad_mem_accum->local_quat_normalize,
        grad_mem_accum->local_quat, num_joint);


    // 4. compute joint local quaternion
    quat_multiply_backward(_stable_pd_mem->parent_quat_inv, _stable_pd_mem->child_quat, grad_mem_accum->local_quat,
        grad_mem_accum->parent_quat_inv, grad_mem_accum->child_quat, num_joint);

    // 3. compute inverse of parent quaternion
    quat_inv_backward_impl(_stable_pd_mem->parent_quat,
        grad_mem_accum->parent_quat_inv, tmp_grad, num_joint);
    for(int i = 0; i < 4 * num_joint; i++) grad_mem_accum->parent_quat[i] += tmp_grad[i];

    // 2. compute child body quaternion
    memset(grad_out_body_quat, 0, sizeof(double) * 4 * num_body);
    _compute_child_body_quaternion_backward(grad_mem_accum->child_quat, grad_out_body_quat, joint_to_child_body, num_joint);
    _compute_parent_body_quaternion_backward(grad_mem_accum->parent_quat, grad_out_body_quat, joint_to_parent_body, num_joint);

    delete[] tmp_grad;
    StablePDControlMemoryNodeDestroy(grad_mem_accum);
}
