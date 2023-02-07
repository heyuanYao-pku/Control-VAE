#pragma once
#include <iostream>
#include <vector>

class TorqueAddHelper
{
	size_t body_cnt = 0;
	size_t joint_cnt = 0;
	std::vector<int> parent_body;
	std::vector<int> child_body;
	std::vector<std::vector<std::pair<int, int> > > grad_map; // pair0: joint index, pair1: count

	std::vector<double> kp;
	std::vector<double> max_len;

public:
	TorqueAddHelper();
	TorqueAddHelper(
		const std::vector<int>& parent_body_,
		const std::vector<int>& child_body_,
		size_t body_cnt_
	);
	~TorqueAddHelper();

public:
	int GetBodyCount() const;
	int GetJointCount() const;
	const std::vector<int> * GetParentBody() const;
	const std::vector<int> * GetChildBody() const;

	const int * get_parent_body_c_ptr() const;
	const int * get_child_body_c_ptr() const;

	const double * get_kp_c_ptr() const;
	const double * get_max_len_c_ptr() const;
	void set_pd_control_param(const std::vector<double> & kp_, const std::vector<double> & max_len_);

public:

	void backward(const double* prev_grad, double* out_grad);

	void add_torque_forward(
		double* body_torque,
		const double* joint_torque);
};

TorqueAddHelper* TorqueAddHelperCreate(
	const std::vector<int>& parent_body_,
	const std::vector<int>& child_body_,
	size_t body_cnt_);

void TorqueAddHelperDelete(TorqueAddHelper* ptr);