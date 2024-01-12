# cython: language_level=3
from EigenWrapper cimport *
from libcpp.vector cimport vector as std_vector
from libcpp.string cimport string as std_string


cdef extern from "PDControlAdd.h" nogil:
    cdef cppclass TorqueAddHelper:
        TorqueAddHelper()
        TorqueAddHelper(std_vector[int]& parent_body_, std_vector[int]& child_body_, int body_cnt_)

        int GetBodyCount() const
        int GetJointCount() const
        const std_vector[int] * GetParentBody() const
        const std_vector[int] * GetChildBody() const

        const int * get_parent_body_c_ptr() const
        const int * get_child_body_c_ptr() const
        const double * get_kp_c_ptr() const
        const double * get_max_len_c_ptr() const
        void set_pd_control_param(const std_vector[double] & kp_, const std_vector[double] & max_len_)

        void backward(const double * prev_grad, double * out_grad)
        void add_torque_forward(double* body_torque, const double * joint_torque)

    TorqueAddHelper* TorqueAddHelperCreate(const std_vector[int]& parent_body_,
        const std_vector[int]& child_body_, size_t body_cnt_)
    void TorqueAddHelperDelete(TorqueAddHelper* ptr)


cdef extern from "MixQuaternion.h" nogil:
    void mix_quaternion(double * quat_input, size_t num, double * result) # The performance of this function is not good..


cdef extern from "QuaternionWithGrad.h" nogil:
    void quat_multiply_single(
        const double * q1,
        const double * q2,
        double * q
    )

    void quat_inv_impl(
        const double * q,
        double * out_q,
        size_t num_quat
    )

    void quat_inv_single(
        const double * q,
        double * out_q
    )

    void quat_multiply_forward(
        const double * q1,
        const double * q2,
        double * q,
        size_t num_quat
    )

    void quat_apply_single(
        const double * q,
        const double * v,
        double * o
    )

    void quat_apply_forward(
        const double * q,
        const double * v,
        double * o,
        size_t num_quat
    )

    # Add by Yulong Zhang
    void quat_apply_forward_one2many(
        const double * q,
        const double * v,
        double * o,
        size_t num_quat
    )

    void flip_quat_by_w_forward_impl(
        const double * q,
        double * q_out,
        size_t num_quat
    )

    void quat_to_vec6d_single(
        const double * q,
        double * vec6d
    )

    void quat_to_vec6d_impl(const double * q, double * vec6d, size_t num_quat)

    void quat_to_matrix_forward_single(
        const double * q,
        double * mat
    )

    void quat_to_matrix_impl(
        const double * q,
        double * mat,
        size_t num_quat
    )

    void six_dim_mat_to_quat_single(
        const double * mat,
        double * quat
    )
    void six_dim_mat_to_quat_impl(
        const double * mat,
        double * q,
        size_t num_quat
    )
    void vector_to_cross_matrix_single(
        const double * vec,
        double * mat
    )

    void vector_to_cross_matrix_impl(
        const double * vec,
        double * mat,
        size_t num_vec
    )

    void quat_to_rotvec_single(
        const double * q,
        double & angle,
        double * rotvec
    )

    void quat_to_rotvec_impl(
        const double * q,
        double * angle,
        double * rotvec,
        size_t num_quat
    )

    void quat_from_rotvec_single(
        const double * rotvec,
        double * q
    )

    void quat_from_rotvec_impl(
        const double * rotvec,
        double * q,
        size_t num_quat
    )

    void quat_from_matrix_single(
        const double * mat,
        double * q
    )

    void quat_from_matrix_impl(
        const double * mat,
        double * q,
        size_t num_quat
    )

    void quat_to_hinge_angle_single(
        const double * q,
        const double * axis,
        double & angle
    )

    void quat_to_hinge_angle_forward(
        const double * q,
        const double * axis,
        double * angle,
        size_t num_quat
    )

    void parent_child_quat_to_hinge_angle(
        const double * quat0,
        const double * quat1,
        const double * init_rel_quat_inv,
        const double * axis,
        double * angle,
        size_t num_quat
    )

    void quat_integrate_impl(
        const double * q,
        const double * omega,
        double dt,
        double * result,
        size_t num_quat
    )

    void vector_normalize_single(
        const double * x,
        size_t ndim,
        double * result
    )

    void quat_integrate_single(
        const double * q,
        const double * omega,
        double dt,
        double * result
    )

    void quat_integrate_impl(
        const double * q,
        const double * omega,
        double dt,
        double * result,
        size_t num_quat
    )

    # Add by Yulong Zhang
    void calc_surface_distance_to_capsule(
        const double * relative_pos,
        size_t ndim,
        double radius,
        double length,
        double * sd,
        double * normal
    )

    void clip_vec_by_norm_forward_single(
        const double * x,
        double min_val,
        double max_val,
        double * result,
        size_t ndim
    )

    void clip_vec_by_length_forward(
        const double * x,
        double max_len,
        double * result,
        size_t ndim
    )

    void clip_vec3_arr_by_length_forward(
        const double * x,
        const double * max_len,
        double * result,
        size_t num_vecs
    )

    void decompose_rotation_single(
        const double * q,
        const double * vb,
        double * result
    )

    void decompose_rotation(
        const double * q,
        const double * v,
        double * result,
        size_t num_quat
    )

    void decompose_rotation_pair_single(
        const double * q,
        const double * vb,
        double * q_a,
        double * q_b
    )

    void decompose_rotation_pair(
        const double * q,
        const double * vb,
        double * q_a,
        double * q_b,
        size_t num_quat
    )

    void decompose_rotation_pair_one2many(
        const double * q,
        const double * vb,
        double * q_a,
        double * q_b,
        size_t num_quat
    )
