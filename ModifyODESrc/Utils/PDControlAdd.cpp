#include "PDControlAdd.h"
#include <algorithm>
#include <iostream>

TorqueAddHelper::TorqueAddHelper()
{

}

TorqueAddHelper::TorqueAddHelper(
	const std::vector<int>& parent_body_,
	const std::vector<int>& child_body_,
	size_t body_cnt_
) : parent_body(parent_body_),
child_body(child_body_),
joint_cnt(static_cast<int>(child_body_.size()))
{
	// body_cnt = joint_cnt + 1;
	this->body_cnt = body_cnt_;
	this->grad_map.resize(body_cnt);
	for (int jidx = 0; jidx < static_cast<int>(joint_cnt); jidx++)
	{
		int pa_idx = parent_body[jidx], ch_idx = child_body[jidx];
		if (pa_idx != -1)
		{
			this->grad_map[pa_idx].push_back(std::make_pair(jidx, -1));
		}
		if (ch_idx != -1)
		{
			this->grad_map[ch_idx].push_back(std::make_pair(jidx, 1));
		}
	}
	for (int bidx = 0; bidx < static_cast<int>(body_cnt); bidx++)
	{
		std::sort(grad_map[bidx].begin(), grad_map[bidx].end());
	}
	//std::cout << "After create " << __func__ << std::endl;
}

TorqueAddHelper::~TorqueAddHelper()
{
	//std::cout << "Deconstruct " << __func__ << std::endl;
}

int TorqueAddHelper::GetBodyCount() const
{
	return body_cnt;
}

int TorqueAddHelper::GetJointCount() const
{
	return joint_cnt;
}

const std::vector<int> * TorqueAddHelper::GetParentBody() const
{
	return &parent_body;
}

const std::vector<int> * TorqueAddHelper::GetChildBody() const
{
	return &child_body;
}

const int * TorqueAddHelper::get_parent_body_c_ptr() const
{
	return parent_body.data();
}


const int * TorqueAddHelper::get_child_body_c_ptr() const
{
	return child_body.data();
}

const double * TorqueAddHelper::get_kp_c_ptr() const
{
	return kp.data();
}

const double * TorqueAddHelper::get_max_len_c_ptr() const
{
	return max_len.data();
}

void TorqueAddHelper::set_pd_control_param(const std::vector<double> & kp_, const std::vector<double> & max_len_)
{
	kp = kp_;
	max_len = max_len_;
}

void TorqueAddHelper::backward(const double* prev_grad, // in shape (num body, 3)
	double* out_grad // in shape (num joint, 3)
)
{
	for (int bidx = 0; bidx < this->body_cnt; bidx++)
	{
		const double* in_ptr = prev_grad + bidx * 3;
		for (const auto& jpair : this->grad_map[bidx])
		{
			int jidx = jpair.first;
			int flag = jpair.second;
			
			double* out_ptr = out_grad + 3 * jidx;
			for (int d = 0; d < 3; d++)
			{
				out_ptr[d] += flag * in_ptr[d];
			}
		}
	}
	// std::cout << "After TorqueAddHelper::" << __func__ << std::endl;
}

void TorqueAddHelper::add_torque_forward(
	double* body_torque,
	const double* joint_torque
)
{
	for (int jidx = 0; jidx < joint_cnt; jidx++)
	{
		const double* j_tau = joint_torque + 3 * jidx;

		// Add to parent body
		int pa_idx = parent_body[jidx], ch_idx = child_body[jidx];
		if (pa_idx != -1)
		{
			double* pa_ptr = body_torque + 3 * pa_idx;
			for (int t = 0; t < 3; t++)
			{
				pa_ptr[t] -= j_tau[t];
			}
		}

		if (ch_idx != -1)
		{
			double* ch_ptr = body_torque + 3 * ch_idx;
			for (int t = 0; t < 3; t++)
			{
				ch_ptr[t] += j_tau[t];
			}
		}
	}
	// std::cout << "After TorqueAddHelper::" << __func__ << std::endl;
}

TorqueAddHelper* TorqueAddHelperCreate(
	const std::vector<int>& parent_body_,
	const std::vector<int>& child_body_,
	size_t body_cnt_)
{
	return new TorqueAddHelper(parent_body_, child_body_, body_cnt_);
}

void TorqueAddHelperDelete(TorqueAddHelper* ptr)
{
	if (ptr != NULL)
	{
		delete ptr;
	}
}