#pragma once
#define FLIP_QUAT_AT_MULTIPLY 0
#include "PDControlAdd.h"
// a differentiable version of quaternion multiply.

void minus_vector(
    const double * x_in,
    double * x_out,
    size_t dim
);

void mat3_set_as_eye_single(
    double * x
);

void mat3_vec3_multiply_single(
    const double * a,
    const double * x,
    double * b
);

void mat3_vec3_multiply_impl(
    const double * a,
    const double * x,
    double * b,
    size_t num_mat
);

void mat3_vec3_multiply_backward_single(
    const double * a,
    const double * x,
    const double * grad_in,
    double * grad_a,
    double * grad_x
);

void mat3_vec3_multiply_backward(
    const double * a,
    const double * x,
    const double * grad_in,
    double * grad_a,
    double * grad_x,
    size_t num_mat
);

void quat_multiply_single(
    const double * q1,
    const double * q2,
    double * q
);

void quat_multiply_forward(
    const double * q1,
    const double * q2,
    double * q,
    size_t num_quat
);

void quat_multiply_backward_single(
    const double * q1,
    const double * q2,
    const double * grad_q, // \frac{\partial L}{\partial q_x, q_y, q_z, q_w}
    double * grad_q1,
    double * grad_q2
);

void quat_multiply_backward(
    const double * q1,
    const double * q2,
    const double * grad_q,
    double * grad_q1,
    double * grad_q2,
    size_t num_quat
);

void quat_apply_single(
    const double * q,
    const double * v,
    double * o
);

void quat_apply_forward(
    const double * q,
    const double * v,
    double * o,
    size_t num_quat
);

// Add by Yulong Zhang
void quat_apply_forward_one2many(
    const double * q,
    const double * v,
    double * o,
    size_t num_quat
);

void quat_apply_backward_single(
    const double * q,
    const double * v,
    const double * o_grad,
    double * q_grad,
    double * v_grad
);

void quat_apply_backward(
    const double * q,
    const double * v,
    const double * o_grad,
    double * q_grad,
    double * v_grad,
    size_t num_quat
);

void flip_quat_by_w_forward_impl(
    const double * q,
    double * q_out,
    size_t num_quat
);

void flip_quat_by_w_backward_impl(
    const double * q,
    const double * grad_in,
    double * grad_out,
    size_t num_quat
);

void quat_to_vec6d_single(
    const double * q,
    double * vec6d
);

void quat_to_vec6d_impl(const double * q, double * vec6d, size_t num_quat);

void quat_to_matrix_forward_single(
    const double * q,
    double * mat
);

// Add by Yulong Zhang
void six_dim_mat_to_quat_single(
    const double * mat,
    double * quat
);

void six_dim_mat_to_quat_impl(
    const double * mat,
    double * q,
    size_t num_quat
);

void quat_to_matrix_impl(
    const double * q,
    double * mat,
    size_t num_quat
);

void quat_to_matrix_backward_single(
    const double * q,
    const double * grad_in,
    double * grad_out
);

void quat_to_matrix_backward(
    const double * q,
    const double * grad_in,
    double * grad_out,
    size_t num_quat
);

void vector_to_cross_matrix_single(
    const double * vec,
    double * mat
);

void vector_to_cross_matrix_impl(
    const double * vec,
    double * mat,
    size_t num_vec
);

void vector_to_cross_matrix_backward_single(
    const double * vec,
    const double * grad_in,
    double * grad_out
);

void vector_to_cross_matrix_backward(
    const double * vec,
    const double * grad_in,
    double * grad_out,
    size_t num_vec
);

void quat_to_rotvec_single(
    const double * q,
    double & angle,
    double * rotvec
);

void quat_to_rotvec_impl(
    const double * q,
    double * angle,
    double * rotvec,
    size_t num_quat
);

void quat_to_rotvec_backward_single(
    const double * q,
    double angle,
    const double * grad_in,
    double * grad_out
);

void quat_to_rotvec_backward(
    const double * q,
    const double * angle,
    const double * grad_in,
    double * grad_out,
    size_t num_quat
);

void quat_from_rotvec_single(
    const double * rotvec,
    double * q
);

void quat_from_rotvec_impl(
    const double * rotvec,
    double * q,
    size_t num_quat
);

void quat_from_rotvec_backward_single(
    const double * rotvec,
    const double * grad_in,
    double * grad_out
);

void quat_from_rotvec_backward_impl(
    const double * rotvec,
    const double * grad_in,
    double * grad_out,
    size_t num_quat
);

void quat_from_matrix_single(
    const double * mat,
    double * q
);

void quat_from_matrix_impl(
    const double * mat,
    double * q,
    size_t num_quat
);

void quat_from_matrix_backward_single(
    const double * mat,
    const double * grad_in,
    double * grad_out
);

void quat_from_matrix_backward_impl(
    const double * mat,
    const double * grad_in,
    double * grad_out,
    size_t num_quat
);

void quat_to_hinge_angle_single(
    const double * q,
    const double * axis,
    double & angle
);

void quat_to_hinge_angle_forward(
    const double * q,
    const double * axis,
    double * angle,
    size_t num_quat
);

void quat_to_hinge_angle_backward_single(
    const double * q,
    const double * axis,
    double grad_in,
    double * grad_out
);

void quat_to_hinge_angle_backward(
    const double * q,
    const double * axis,
    const double * grad_in,
    double * grad_out,
    size_t num_quat
);

void quat_inv_single(
    const double * q,
    double * out_q
);

void quat_inv_impl(
    const double * q,
    double * out_q,
    size_t num_quat
);

void quat_inv_backward_single(
    const double * q,
    const double * grad_in,
    double * grad_out
);

void quat_inv_backward_impl(
    const double * q,
    const double * grad_in,
    double * grad_out,
    size_t num_quat
);

void parent_child_quat_to_hinge_angle_single_func(
    const double * quat0,
    const double * quat1,
    const double * init_rel_quat_inv,
    const double * axis,
    double & angle,
    double * quat0_inv,
    double * dq,
    double * dquat
);

void parent_child_quat_to_hinge_angle_single(
    const double * quat0,
    const double * quat1,
    const double * init_rel_quat_inv,
    const double * axis,
    double & angle
);

void parent_child_quat_to_hinge_angle(
    const double * quat0,
    const double * quat1,
    const double * init_rel_quat_inv,
    const double * axis,
    double * angle,
    size_t num_quat
);

void parent_child_quat_to_hinge_angle_backward_single(
    const double * quat0,
    const double * quat1,
    const double * init_rel_quat_inv,
    const double * axis,
    double grad_in,
    double * quat0_grad,
    double * quat1_grad
);

void parent_child_quat_to_hinge_angle_backward(
    const double * quat0,
    const double * quat1,
    const double * init_rel_quat_inv,
    const double * axis,
    const double * grad_in,
    double * quat0_grad,
    double * quat1_grad,
    size_t num_quat
);

void vector_normalize_single(
    const double * x,
    size_t ndim,
    double * result
);

void vector_normalize_backward_single(
    const double * x,
    size_t ndim,
    const double * grad_in,
    double * grad_out
);

void quat_integrate_single(
    const double * q,
    const double * omega,
    double dt,
    double * result
);

void quat_integrate_impl(
    const double * q,
    const double * omega,
    double dt,
    double * result,
    size_t num_quat
);

void quat_integrate_backward_single(
    const double * q,
    const double * omega,
    double dt,
    const double * grad_in,
    double * q_grad,
    double * omega_grad
);

void quat_integrate_backward(
    const double * q,
    const double * omega,
    double dt,
    const double * grad_in,
    double * q_grad,
    double * omega_grad,
    size_t num_quat
);

// Add by Yulong Zhang
void calc_surface_distance_to_capsule_single(
    const double * relative_pos,
    const double radius,
    const double length,
    double & sd,
    double * normal
);

void calc_surface_distance_to_capsule(
    const double * relative_pos,
    size_t ndim,
    const double radius,
    const double length,
    double * sd,
    double * normal
);

void clip_vec_by_norm_forward_single(
    const double * x,
    double min_val,
    double max_val,
    double * result,
    size_t ndim
);

void clip_vec_by_length_forward(
    const double * x,
    double max_len,
    double * result,
    size_t ndim
);

void clip_vec3_arr_by_length_forward(
    const double * x,
    const double * max_len,
    double * result,
    size_t num_vecs
);

void clip_vec_by_length_backward(
    const double * x,
    double max_len,
    const double * grad_in,
    double * grad_out,
    size_t ndim
);

void clip_vec3_arr_by_length_backward(
    const double * x,
    const double * max_len,
    const double * grad_in,
    double * grad_out,
    size_t num_vecs
);

void decompose_rotation_single(
    const double * q,
    const double * vb,
    double * result
);

void decompose_rotation(
    const double * q,
    const double * v,
    double * result,
    size_t num_quat
);

void decompose_rotation_backward_single(
    const double * q,
    const double * v,
    const double * grad_in,
    double * grad_q,
    double * grad_v
);

void decompose_rotation_backward(
    const double * q,
    const double * v,
    const double * grad_in,
    double * grad_q,
    double * grad_v,
    size_t num_quat
);

void decompose_rotation_pair_single(
    const double * q,
    const double * vb,
    double * q_a,
    double * q_b
);

void decompose_rotation_pair(
    const double * q,
    const double * vb,
    double * q_a,
    double * q_b,
    size_t num_quat
);

void decompose_rotation_pair_one2many(
    const double * q,
    const double * vb,
    double * q_a,
    double * q_b,
    size_t num_quat
);

struct StablePDControlMemoryNode{
    double * parent_quat = nullptr;
    double * child_quat = nullptr;
    double * parent_quat_inv = nullptr;
    double * local_quat = nullptr;
    double * local_quat_normalize = nullptr;

    double * inv_local_quat = nullptr;
    double * delta_quat = nullptr;
    double * rotvec = nullptr;
    double * angle = nullptr;
    double * torque = nullptr;
    double * clip_torque = nullptr;
    double * global_joint_torque = nullptr;
    double * global_body_torque = nullptr;

    void mem_alloc(size_t num_joint, size_t num_body);
    void mem_dealloc();
    void zero(size_t num_joint, size_t num_body);
};

StablePDControlMemoryNode * StablePDControlMemoryNodeCreate(size_t num_joint, size_t num_body);

void StablePDControlMemoryNodeDestroy(StablePDControlMemoryNode * node);

void stable_pd_control_forward(
    const double * body_quat,
    const double * target_quat,
    double * global_body_torque,
    TorqueAddHelper * helper,
    StablePDControlMemoryNode * _stable_pd_mem
);

void stable_pd_control_backward(
    const double * body_quat,
    const double * target_quat,
    const double * grad_in,
    double * grad_out_body_quat,
    double * grad_out_target_quat,
    TorqueAddHelper * helper,
    StablePDControlMemoryNode * _stable_pd_mem
);
