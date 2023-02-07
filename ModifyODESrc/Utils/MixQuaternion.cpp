# include <iostream>
# include <Eigen/Eigen>

void mix_quaternion(double * quat_input, size_t num, double * result)
{
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> quat_in(quat_input, num, 4);
	Eigen::Quaterniond res(quat_in.row(0).data());
    res.normalize();
	for(int i = 1; i < num; i++)
	{
		Eigen::Quaterniond quat_i(quat_in.row(i).data());
        quat_i.normalize();
        if (res.dot(quat_i) < 0)
        {
            quat_i = Eigen::Quaterniond(-quat_i.coeffs());
        }

		double coef = 1.0 - 1.0 / (i + 1);
		res = res.slerp(coef, quat_i).normalized();
	}
	Eigen::Map<Eigen::Quaterniond> res_map(result);
	res_map = res;
}


static void test_slerp()
{
	Eigen::Quaterniond q1(Eigen::Quaterniond::UnitRandom());
	Eigen::Quaterniond q2(Eigen::Quaterniond::UnitRandom());
	std::cout << "q1 \n" << q1.coeffs() << std::endl;
	std::cout << "q2 \n" << q2.coeffs() << std::endl;

	std::cout << "slerp t = 0 \n" << q1.slerp(0, q2).coeffs() << std::endl;
}

static void test_func()
{
	// test_slerp();
	int num = 10;
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> quat_in(num, 4);
	for(int i=0; i<num; i++)
	{
		Eigen::Quaterniond rand_q = Eigen::Quaterniond::UnitRandom();
		quat_in.row(i) = rand_q.coeffs();
		//std::cout << quat_in.row(i) << " w = " << rand_q.w() << std::endl;
	}
	// for(int i=0; i<num; i++)
	// {
	// 	for(int j=0; j<4; j++)
	// 	{
	// 		std::cout << quat_in.data()[4 * i + j] << " ";
	// 	}
	// 	std::cout << std::endl;
	// }
	double result[4] = {0, 0, 0, 0};
	mix_quaternion(quat_in.data(), num, result);
	
	std::cout << "print result" << std::endl;
	for(int i=0; i<4; i++)
	{
		std::cout << result[i] << " ";
	}
	std::cout << std::endl;
}
