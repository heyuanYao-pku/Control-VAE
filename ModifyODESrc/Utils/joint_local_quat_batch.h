#pragma once
#include <ode/ode.h>

void ode_quat_to_scipy(dReal* q_ode);

void scipy_quat_to_ode(dReal* q_scipy);

void ode_quat_inv(dReal* ode_quat);

void quat_arr_from_ode_to_scipy(dReal* qs, int count);

void quat_arr_from_scipy_to_ode(dReal* qs, int count);

dReal quat_dot(const dReal* q0, const dReal* q1);

void minus_quat(dReal* q);

void flip_ode_quat_by_w(dReal* ode_quat);

void ode_quat_apply(const dReal* quat_ode, const dReal* vec_in, dReal* vec_out);

// modified from python library scipy.spatial.transform.Rotation, with scipy version == 1.5.4
void ode_quat_to_axis_angle(const dReal* ode_quat_input, dReal* result);

void get_joint_local_quat_batch(
    dJointID* joints,
    int joint_count,
    dReal* parent_qs,
    dReal* child_qs,
    dReal* local_qs,
    dReal* parent_qs_inv,
    int convert_to_scipy);

dReal compute_total_power_by_global(dJointID * joints, int joint_count, const dReal * global_joint_torques);
dReal compute_total_power(dJointID * joints, int joint_count, const dReal * joint_torques);

// As (stable) PD controller in python is too slow (by profile), rewrite it in c++, then call it via cython
// TODO: compare with result PD controller result in python version.
void pd_control_batch(
    dJointID* joints,
    int joint_count,
    const dReal* input_target_local_qs,
    const dReal* kps,
    const dReal* kds,
    const dReal* torque_limits,
    dReal* local_res_joint_torques,
    dReal* global_res_joint_torques,
    int input_in_scipy
);
