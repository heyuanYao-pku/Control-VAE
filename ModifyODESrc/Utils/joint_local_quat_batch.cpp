#include "joint_local_quat_batch.h"

#include <algorithm>
#include <cmath>
#include <iostream>

void ode_quat_to_scipy(dReal* q_ode)
{
    dReal res_quat[4];
    res_quat[0] = q_ode[1];
    res_quat[1] = q_ode[2];
    res_quat[2] = q_ode[3];
    res_quat[3] = q_ode[0];
    std::copy(res_quat, res_quat + 4, q_ode);
}

void scipy_quat_to_ode(dReal* q_scipy)
{
    dReal q_ode[4];
    q_ode[0] = q_scipy[3];
    q_ode[1] = q_scipy[0];
    q_ode[2] = q_scipy[1];
    q_ode[3] = q_scipy[2];
    std::copy(q_ode, q_ode + 4, q_scipy);
}

void ode_quat_inv(dReal* ode_quat)
{
    // (w, x, y, z) in quaternion of ode.
    for (int i = 1; i < 4; i++)
    {
        ode_quat[i] = -ode_quat[i];
    }
}

void quat_arr_from_ode_to_scipy(dReal* qs, int count)
{
    for (int i = 0; i < count; i++)
    {
        ode_quat_to_scipy(qs + 4 * i);
    }
}

void quat_arr_from_scipy_to_ode(dReal* qs, int count)
{
    for (int i = 0; i < count; i++)
    {
        scipy_quat_to_ode(qs + 4 * i);
    }
}

dReal quat_dot(const dReal* q0, const dReal* q1)
{
    dReal res = 0;
    for (int i = 0; i < 4; i++)
    {
        res += q0[i] * q1[i];
    }
    return res;
}

void minus_quat(dReal* q)
{
    for (int i = 0; i < 4; i++)
    {
        q[i] = -q[i];
    }
}

void flip_ode_quat_by_w(dReal* ode_quat)
{
    if (ode_quat[0] < 0)
    {
        minus_quat(ode_quat);
    }
}

void ode_quat_apply(const dReal* quat_ode, const dReal* vec_in, dReal* vec_out)
{
    dQuaternion quat_vec_in;
    quat_vec_in[0] = 0;
    std::copy(vec_in, vec_in + 3, quat_vec_in + 1);
    dQuaternion qv, qvq_inv;
    dQMultiply0(qv, quat_ode, quat_vec_in);
    dQMultiply2(qvq_inv, qv, quat_ode);

    std::copy(qvq_inv + 1, qvq_inv + 4, vec_out);
}

// modified from python library scipy.spatial.transform.Rotation, with scipy version == 1.5.4
void ode_quat_to_axis_angle(const dReal * ode_quat_input, dReal * result)
{
    dReal ode_quat[4];
    const dReal* quat_xyz = ode_quat + 1;
    std::copy(ode_quat_input, ode_quat_input + 4, ode_quat);
    flip_ode_quat_by_w(ode_quat);
    dReal axis_len = dCalcVectorLength3(quat_xyz), w = ode_quat[0];
    dReal angle = 2 * std::atan2(axis_len, w);
    dReal scale = 1.0;
    if (std::abs(angle) <= 1e-3)
    {
        dReal angle_pow_2 = angle * angle;
        dReal angle_pow_4 = angle_pow_2 * angle_pow_2;
        scale = 2 + angle * angle / 12 + (7.0 / 2880) * angle_pow_4;
    }
    else
    {
        scale = angle / std::sin(0.5 * angle);
    }
    for (int i = 0; i < 3; i++)
    {
        result[i] = scale * quat_xyz[i];
    }
}

void get_joint_local_quat_batch(
    dJointID* joints,
    int joint_count,
    dReal* parent_qs,
    dReal* child_qs,
    dReal* local_qs,
    dReal* parent_qs_inv,
    int convert_to_scipy = 1)
{
    const dReal* const_quat_0 = NULL;
    const dReal* const_quat_1 = NULL;

    dReal * parent_quat = NULL;
    dReal* child_quat = NULL;
    dReal* local_quat = NULL;
    dReal* parent_quat_inv = NULL;

    dBodyID body0 = NULL, body1 = NULL;
    dJointID jid = NULL;
    dQuaternion unit_quat_ode;
    dQSetIdentity(unit_quat_ode);

    // dReal quat0_norm = 1, quat1_norm = 1;
    for (int i = 0; i < joint_count; i++)
    {
        jid = joints[i];
        body0 = dJointGetBody(jid, 0);
        body1 = dJointGetBody(jid, 1);
        const_quat_0 = dBodyGetQuaternion(body0);

        if (body1 != NULL)
        {
            const_quat_1 = dBodyGetQuaternion(body1);
        }
        else
        {
            const_quat_1 = unit_quat_ode;
        }

        child_quat = child_qs + 4 * i;
        parent_quat = parent_qs + 4 * i;
        local_quat = local_qs + 4 * i;
        parent_quat_inv = parent_qs_inv + 4 * i;

        std::copy(const_quat_0, const_quat_0 + 4, child_quat);
        std::copy(const_quat_1, const_quat_1 + 4, parent_quat);
        dNormalize4(child_quat);
        dNormalize4(parent_quat);

        // Compute local quaternion
        dQMultiply1(local_quat, parent_quat, child_quat);
        dNormalize4(local_quat);

        // Compute inverse of parent quaternion
        std::copy(parent_quat, parent_quat + 4, parent_quat_inv);
        ode_quat_inv(parent_quat_inv);
    }

    if (convert_to_scipy)
    {
        quat_arr_from_ode_to_scipy(parent_qs, joint_count);
        quat_arr_from_ode_to_scipy(child_qs, joint_count);
        quat_arr_from_ode_to_scipy(local_qs, joint_count);
        quat_arr_from_ode_to_scipy(parent_qs_inv, joint_count);
    }
}

dReal compute_total_power_by_global(dJointID * joints, int joint_count, const dReal * global_joint_torques)
{
    dReal total_power = dReal(0.0);
    const dReal * joint_torque = global_joint_torques;
    dBodyID body0 = NULL;
    dJointID jid = NULL;
    const dReal * angvel_0;
    for(int jidx=0; jidx<joint_count; jidx++)
    {
        jid = joints[jidx];
        body0 = dJointGetBody(jid, 0);
        joint_torque = global_joint_torques + 3 * jidx;
        angvel_0 = dBodyGetAngularVel(body0); // 0 is child, and 1 is parent
        dReal dot_res = dReal(0.0);
        for(int i=0; i<3; i++)
        {
            dot_res += joint_torque[i] * angvel_0[i];
        }
        total_power += abs(dot_res);
    }
    return total_power;
}

// compute character power by local coordinate
dReal compute_total_power(dJointID * joints, int joint_count, const dReal * joint_torques)
{
    dReal total_power = dReal(0.0);
    const dReal * joint_torque = joint_torques;
    dBodyID body0 = NULL, body1 = NULL;
    dJointID jid = NULL;
    const dReal * angvel_0, * angvel_1;
    dVector3 zero_vector3, joint_angvel;
    memset(zero_vector3, 0, sizeof(zero_vector3));
    for(int jidx=0; jidx<joint_count; jidx++)
    {
        jid = joints[jidx];
        body0 = dJointGetBody(jid, 0);
        body1 = dJointGetBody(jid, 1);
        joint_torque = joint_torques + 3 * jidx;
        angvel_0 = dBodyGetAngularVel(body0); // 0 is child, and 1 is parent
        if (body1 != NULL)
        {
            angvel_1 = dBodyGetAngularVel(body1);
        }
        else
        {
            angvel_1 = zero_vector3;
        }
        dSubtractVectors3(joint_angvel, angvel_0, angvel_1);
        dReal dot_res = dReal(0.0);
        for(int i=0; i<3; i++)
        {
            dot_res += joint_torque[i] * joint_angvel[i];
        }
        total_power += abs(dot_res);
    }
    return total_power;
}

// As (stable) PD controller in python is too slow (by profile), rewrite it in c++, then call it via cython/
// TODO: compare with result PD controller result in python version.
void pd_control_batch(
    dJointID * joints,
    int joint_count,
    const dReal * input_target_local_qs,
    const dReal * kps,
    const dReal * kds,
    const dReal * torque_limits,
    dReal * local_res_joint_torques,
    dReal * global_res_joint_torques,
    int input_in_scipy = 1
)
{
    // alloc memory for calc joint local quaternion
    const int qs_total_size = 4 * joint_count;
    dReal* parent_qs = new dReal[qs_total_size];
    dReal* child_qs = new dReal[qs_total_size];
    dReal* local_qs = new dReal[qs_total_size];
    dReal* parent_qs_inv = new dReal[qs_total_size];
    std::fill(parent_qs, parent_qs + qs_total_size, 0);
    std::fill(child_qs, child_qs + qs_total_size, 0);
    std::fill(local_qs, local_qs + qs_total_size, 0);
    std::fill(parent_qs_inv, parent_qs_inv + qs_total_size, 0);

    // calc joint local quaternion
    get_joint_local_quat_batch(
        joints,
        joint_count,
        parent_qs,
        child_qs,
        local_qs,
        parent_qs_inv,
        0);

    // convert target pose to ode format.
    dReal* target_local_qs = new dReal[qs_total_size];
    std::copy(input_target_local_qs, input_target_local_qs + qs_total_size, target_local_qs);
    if (input_in_scipy)
    {
        quat_arr_from_scipy_to_ode(target_local_qs, joint_count);
    }

    dQuaternion delta_quat;
    dReal* parent_q = NULL;
    dReal* local_q = NULL;
    const dReal* tar_local_q = NULL;
    dReal axis_angle[3];
    dReal* local_torque = NULL;
    dReal* global_torque = NULL;

    for (int jidx = 0; jidx < joint_count; jidx++)
    {
        // compute difference between joint local quaternion and target pose
        parent_q = parent_qs + 4 * jidx;
        local_q = local_qs + 4 * jidx;
        tar_local_q = target_local_qs + 4 * jidx;
        local_torque = local_res_joint_torques + 3 * jidx;
        global_torque = global_res_joint_torques + 3 * jidx;

        if (quat_dot(local_q, tar_local_q) < 0)
        {
            minus_quat(local_q);
        }

        dQMultiply2(delta_quat, tar_local_q, local_q);
        dNormalize4(delta_quat);
        // convert quaternion to axis angle format.
        ode_quat_to_axis_angle(delta_quat, axis_angle);
        for (int dim = 0; dim < 3; dim++)
        {
            local_torque[dim] = kps[jidx] * axis_angle[dim];
        }

        if (kds != NULL) // TODO: for PD controller
        {
            std::cout << "Warning: Add kd not implemented now." << std::endl;
        }

        // clip joint torque
        dReal torque_len = dCalcVectorLength3(local_torque);
        if (torque_len < 1e-10)
        {
            torque_len = 1;
        }
        dReal ratio = torque_len;
        if (ratio < -torque_limits[jidx])
        {
            ratio = -torque_limits[jidx];
        }
        else if (ratio > torque_limits[jidx])
        {
            ratio = torque_limits[jidx];
        }
        for (int dim = 0; dim < 3; dim++)
        {
            local_torque[dim] = local_torque[dim] / torque_len * ratio;
        }

        // convert joint torque to global coordinate. That is, apply parent rotation to local torque
        ode_quat_apply(parent_q, local_torque, global_torque);
    }

    delete[] parent_qs;
    delete[] child_qs;
    delete[] local_qs;
    delete[] parent_qs_inv;

    delete[] target_local_qs;
}
