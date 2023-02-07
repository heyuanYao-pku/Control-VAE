#pragma once
#include <Eigen/Dense>


struct QuaternionResultf
{
	float x;
	float y;
	float z;
	float w;
};

struct EulerResultf
{
	float xAngle;
	float yAngle;
	float zAngle;
	int unique;
};

template<class RealType = float, std::enable_if_t<std::is_floating_point<RealType>::value, bool> = true>
struct EulerResult
{
	RealType xAngle;
	RealType yAngle;
	RealType zAngle;
	int unique;
};

template<class RealType = float, std::enable_if_t<std::is_floating_point<RealType>::value, bool> = true>
Eigen::Matrix<RealType, 3, 3> EulerMakerXYZ(RealType xAngle, RealType yAngle, RealType zAngle)
{
	return (Eigen::AngleAxis<RealType>(xAngle, Eigen::Matrix<RealType, 3, 1>::UnitX()) *
		Eigen::AngleAxis<RealType>(yAngle, Eigen::Matrix<RealType, 3, 1>::UnitY()) *
		Eigen::AngleAxis<RealType>(zAngle, Eigen::Matrix<RealType, 3, 1>::UnitZ())).matrix();
}

template<class RealType = float, std::enable_if_t<std::is_floating_point<RealType>::value, bool> = true>
Eigen::Matrix<RealType, 3, 3> EulerMakerXZY(RealType xAngle, RealType zAngle, RealType yAngle)
{
	return (Eigen::AngleAxis<RealType>(xAngle, Eigen::Matrix<RealType, 3, 1>::UnitX()) *
		Eigen::AngleAxis<RealType>(zAngle, Eigen::Matrix<RealType, 3, 1>::UnitZ()) *
		Eigen::AngleAxis<RealType>(yAngle, Eigen::Matrix<RealType, 3, 1>::UnitY())).matrix();
}

template<class RealType = float, std::enable_if_t<std::is_floating_point<RealType>::value, bool> = true>
Eigen::Matrix<RealType, 3, 3> EulerMakerYXZ(RealType yAngle, RealType xAngle, RealType zAngle)
{
	return (Eigen::AngleAxis<RealType>(yAngle, Eigen::Matrix<RealType, 3, 1>::UnitY()) *
		Eigen::AngleAxis<RealType>(xAngle, Eigen::Matrix<RealType, 3, 1>::UnitX()) *
		Eigen::AngleAxis<RealType>(zAngle, Eigen::Matrix<RealType, 3, 1>::UnitZ())).matrix();
}

template<class RealType = float, std::enable_if_t<std::is_floating_point<RealType>::value, bool> = true>
Eigen::Matrix<RealType, 3, 3> EulerMakerYZX(RealType yAngle, RealType zAngle, RealType xAngle)
{
	return (Eigen::AngleAxis<RealType>(yAngle, Eigen::Matrix<RealType, 3, 1>::UnitY()) *
		Eigen::AngleAxis<RealType>(zAngle, Eigen::Matrix<RealType, 3, 1>::UnitZ()) *
		Eigen::AngleAxis<RealType>(xAngle, Eigen::Matrix<RealType, 3, 1>::UnitX())).matrix();
}

template<class RealType = float, std::enable_if_t<std::is_floating_point<RealType>::value, bool> = true>
Eigen::Matrix<RealType, 3, 3> EulerMakerZXY(RealType zAngle, RealType xAngle, RealType yAngle)
{
	return (Eigen::AngleAxis<RealType>(zAngle, Eigen::Matrix<RealType, 3, 1>::UnitZ()) *
		Eigen::AngleAxis<RealType>(xAngle, Eigen::Matrix<RealType, 3, 1>::UnitX()) *
		Eigen::AngleAxis<RealType>(yAngle, Eigen::Matrix<RealType, 3, 1>::UnitY())).matrix();
}

template<class RealType = float, std::enable_if_t<std::is_floating_point<RealType>::value, bool> = true>
Eigen::Matrix<RealType, 3, 3> EulerMakerZYX(RealType zAngle, RealType yAngle, RealType xAngle)
{
	return (Eigen::AngleAxis<RealType>(zAngle, Eigen::Matrix<RealType, 3, 1>::UnitZ()) *
		Eigen::AngleAxis<RealType>(yAngle, Eigen::Matrix<RealType, 3, 1>::UnitY()) *
		Eigen::AngleAxis<RealType>(xAngle, Eigen::Matrix<RealType, 3, 1>::UnitX())).matrix();
}

template<class RealType = float, std::enable_if_t<std::is_floating_point<RealType>::value, bool> = true>
Eigen::Matrix<RealType, 3, 3> * Matrix3MakeEulerXYZ(RealType xAngle, RealType yAngle, RealType zAngle)
{
	Eigen::Matrix<RealType, 3, 3> * m = new Eigen::Matrix<RealType, 3, 3>();
	*m = EulerMakerXYZ(xAngle, yAngle, zAngle);
	return m;
}

template<class RealType = float, std::enable_if_t<std::is_floating_point<RealType>::value, bool> = true>
Eigen::Matrix<RealType, 3, 3>* Matrix3MakeEulerXZY(RealType xAngle, RealType zAngle, RealType yAngle)
{
	Eigen::Matrix<RealType, 3, 3> * m = new Eigen::Matrix<RealType, 3, 3>();
	*m = EulerMakerXZY(xAngle, zAngle, yAngle);
	return m;
}

template<class RealType = float, std::enable_if_t<std::is_floating_point<RealType>::value, bool> = true>
Eigen::Matrix<RealType, 3, 3> * Matrix3MakeEulerYXZ(RealType yAngle, RealType xAngle, RealType zAngle)
{
	Eigen::Matrix<RealType, 3, 3> * m = new Eigen::Matrix<RealType, 3, 3>();
	*m = EulerMakerYXZ(yAngle, xAngle, zAngle);
	return m;
}

template<class RealType = float, std::enable_if_t<std::is_floating_point<RealType>::value, bool> = true>
Eigen::Matrix<RealType, 3, 3> * Matrix3MakeEulerYZX(RealType yAngle, RealType zAngle, RealType xAngle)
{
	Eigen::Matrix<RealType, 3, 3> * m = new Eigen::Matrix<RealType, 3, 3>();
	*m = EulerMakerYZX(yAngle, zAngle, xAngle);
	return m;
}

template<class RealType = float, std::enable_if_t<std::is_floating_point<RealType>::value, bool> = true>
Eigen::Matrix<RealType, 3, 3> * Matrix3MakeEulerZXY(RealType zAngle, RealType xAngle, RealType yAngle)
{
	Eigen::Matrix<RealType, 3, 3>* m = new Eigen::Matrix<RealType, 3, 3>();
	*m = EulerMakerZXY(zAngle, xAngle, yAngle);
	return m;
}

template<class RealType = float, std::enable_if_t<std::is_floating_point<RealType>::value, bool> = true>
Eigen::Matrix<RealType, 3, 3> * Matrix3MakeEulerZYX(RealType zAngle, RealType yAngle, RealType xAngle)
{
	Eigen::Matrix<RealType, 3, 3>* m = new Eigen::Matrix<RealType, 3, 3>();
	*m = EulerMakerZYX(zAngle, yAngle, xAngle);
	return m;
}

template<class RealType = float, class ResultType = EulerResultf, std::enable_if_t<std::is_floating_point<RealType>::value, bool> = true>
ResultType Matrix3ExtractEulerXYZ(const Eigen::Matrix<RealType, 3, 3>* ptr)
{
	// +-           -+   +-                                        -+
			// | r00 r01 r02 |   |  cy*cz           -cy*sz            sy    |
			// | r10 r11 r12 | = |  cz*sx*sy+cx*sz   cx*cz-sx*sy*sz  -cy*sx |
			// | r20 r21 r22 |   | -cx*cz*sy+sx*sz   cz*sx+cx*sy*sz   cx*cy |
			// +-           -+   +-                                        -+
			// float res.xAngle, res.yAngle, res.zAngle;
	ResultType res;
	if ((*ptr)(0, 2) < 1)
	{
		if ((*ptr)(0, 2) > -1)
		{
			// y_angle = asin(r02)
			// x_angle = atan2(-r12,r22)
			// z_angle = atan2(-r01,r00)
			res.yAngle = asin((*ptr)(0, 2));
			res.xAngle = atan2(-(*ptr)(1, 2), (*ptr)(2, 2));
			res.zAngle = atan2(-(*ptr)(0, 1), (*ptr)(0, 0));
			res.unique = 1;
			// res.unique = 1;
		}
		else
		{
			// y_angle = -pi/2
			// z_angle - x_angle = atan2(r10,r11)
			// WARNING.  The solution is not unique.  Choosing z_angle = 0.

			res.yAngle = -static_cast<RealType>(M_PI_2);
			res.xAngle = -atan2((*ptr)(1, 0), (*ptr)(1, 1));
			res.zAngle = 0;
			res.unique = 0;
			// res.unique = 0;
		}
	}
	else
	{
		// y_angle = +pi/2
		// z_angle + x_angle = atan2(r10,r11)
		// WARNING.  The solutions is not unique.  Choosing z_angle = 0.

		res.yAngle = static_cast<RealType>(M_PI_2);
		res.xAngle = atan2((*ptr)(1, 0), (*ptr)(1, 1));
		res.zAngle = 0;
		res.unique = 0;
		// res.unique = 0;
	}
	return res;
}

template<class RealType, class ResultType, std::enable_if_t<std::is_floating_point<RealType>::value, bool> = true>
ResultType Matrix3ExtractEulerXZY(const Eigen::Matrix<RealType, 3, 3> * ptr)
{
	// +-           -+   +-                                        -+
	// | r00 r01 r02 |   |  cy*cz           -sz      cz*sy          |
	// | r10 r11 r12 | = |  sx*sy+cx*cy*sz   cx*cz  -cy*sx+cx*sy*sz |
	// | r20 r21 r22 |   | -cx*sy+cy*sx*sz   cz*sx   cx*cy+sx*sy*sz |
	// +-           -+   +-                                        -+
	ResultType res;
	if ((*ptr)(0, 1) < 1)
	{
		if ((*ptr)(0, 1) > -1)
		{
			// z_angle = asin(-r01)
			// x_angle = atan2(r21,r11)
			// y_angle = atan2(r02,r00)
			res.zAngle = asin(-(*ptr)(0, 1));
			res.xAngle = atan2((*ptr)(2, 1), (*ptr)(1, 1));
			res.yAngle = atan2((*ptr)(0, 2), (*ptr)(0, 0));
			res.unique = 1;
		}
		else
		{
			// z_angle = +pi/2
			// y_angle - x_angle = atan2(-r20,r22)
			// WARNING.  The solution is not unique.  Choosing y_angle = 0.
			res.zAngle = static_cast<float>(M_PI_2);
			res.xAngle = -atan2(-(*ptr)(2, 0), (*ptr)(2, 2));
			res.yAngle = 0;
			res.unique = 0;
		}
	}
	else
	{
		// z_angle = -pi/2
		// y_angle + x_angle = atan2(-r20,r22)
		// WARNING.  The solution is not unique.  Choosing y_angle = 0.
		res.zAngle = -static_cast<float>(M_PI_2);
		res.xAngle = atan2(-(*ptr)(2, 0), (*ptr)(2, 2));
		res.yAngle = 0;
		res.unique = 0;
	}
	return res;
}

template<class RealType, class ResultType, std::enable_if_t<std::is_floating_point<RealType>::value, bool> = true>
ResultType Matrix3ExtractEulerYXZ(const Eigen::Matrix<RealType, 3, 3> * ptr)
{
	// +-           -+   +-                                       -+
	// | r00 r01 r02 |   |  cy*cz+sx*sy*sz  cz*sx*sy-cy*sz   cx*sy |
	// | r10 r11 r12 | = |  cx*sz           cx*cz           -sx    |
	// | r20 r21 r22 |   | -cz*sy+cy*sx*sz  cy*cz*sx+sy*sz   cx*cy |
	// +-           -+   +-                                       -+
	ResultType res;
	if ((*ptr)(1, 2) < 1)
	{
		if ((*ptr)(1, 2) > -1)
		{
			// x_angle = asin(-r12)
			// y_angle = atan2(r02,r22)
			// z_angle = atan2(r10,r11)
			res.xAngle = asin(-(*ptr)(1, 2));
			res.yAngle = atan2((*ptr)(0, 2), (*ptr)(2, 2));
			res.zAngle = atan2((*ptr)(1, 0), (*ptr)(1, 1));
			res.unique = 1;
		}
		else
		{
			// x_angle = +pi/2
			// z_angle - y_angle = atan2(-r01,r00)
			// WARNING.  The solution is not unique.  Choosing z_angle = 0.
			res.xAngle = static_cast<float>(M_PI_2);
			res.yAngle = -atan2(-(*ptr)(0, 1), (*ptr)(0, 0));
			res.zAngle = 0;
			res.unique = 0;
		}
	}
	else
	{
		// x_angle = -pi/2
		// z_angle + y_angle = atan2(-r01,r00)
		// WARNING.  The solution is not unique.  Choosing z_angle = 0.
		res.xAngle = -static_cast<float>(M_PI_2);
		res.yAngle = atan2(-(*ptr)(0, 1), (*ptr)(0, 0));
		res.zAngle = 0;
		res.unique = 0;
	}
	return res;
}

template<class RealType, class ResultType, std::enable_if_t<std::is_floating_point<RealType>::value, bool> = true>
ResultType Matrix3ExtractEulerYZX(const Eigen::Matrix<RealType, 3, 3> * ptr)
{
	// +-           -+   +-                                       -+
	// | r00 r01 r02 |   |  cy*cz  sx*sy-cx*cy*sz   cx*sy+cy*sx*sz |
	// | r10 r11 r12 | = |  sz     cx*cz           -cz*sx          |
	// | r20 r21 r22 |   | -cz*sy  cy*sx+cx*sy*sz   cx*cy-sx*sy*sz |
	// +-           -+   +-                                       -+
	ResultType res;
	if ((*ptr)(1, 0) < 1)
	{
		if ((*ptr)(1, 0) > -1)
		{
			// z_angle = asin(r10)
			// y_angle = atan2(-r20,r00)
			// x_angle = atan2(-r12,r11)
			res.zAngle = asin((*ptr)(1, 0));
			res.yAngle = atan2(-(*ptr)(2, 0), (*ptr)(0, 0));
			res.xAngle = atan2(-(*ptr)(1, 2), (*ptr)(1, 1));
			res.unique = 1;
		}
		else
		{
			// z_angle = -pi/2
			// x_angle - y_angle = atan2(r21,r22)
			// WARNING.  The solution is not unique.  Choosing x_angle = 0.
			res.zAngle = -static_cast<float>(M_PI_2);
			res.yAngle = -atan2((*ptr)(2, 1), (*ptr)(2, 2));
			res.xAngle = 0;
			res.unique = 0;
		}
	}
	else
	{
		// z_angle = +pi/2
		// x_angle + y_angle = atan2(r21,r22)
		// WARNING.  The solution is not unique.  Choosing x_angle = 0.
		res.zAngle = static_cast<float>(M_PI_2);
		res.yAngle = atan2((*ptr)(2, 1), (*ptr)(2, 2));
		res.xAngle = 0;
		res.unique = 0;
	}
	return res;
}

template<class RealType, class ResultType, std::enable_if_t<std::is_floating_point<RealType>::value, bool> = true>
ResultType Matrix3ExtractEulerZXY(const Eigen::Matrix<RealType, 3, 3> * ptr)
{
	// +-           -+   +-                                        -+
	// | r00 r01 r02 |   |  cy*cz-sx*sy*sz  -cx*sz   cz*sy+cy*sx*sz |
	// | r10 r11 r12 | = |  cz*sx*sy+cy*sz   cx*cz  -cy*cz*sx+sy*sz |
	// | r20 r21 r22 |   | -cx*sy            sx      cx*cy          |
	// +-           -+   +-                                        -+
	ResultType res;
	if ((*ptr)(2, 1) < 1)
	{
		if ((*ptr)(2, 1) > -1)
		{
			// x_angle = asin(r21)
			// z_angle = atan2(-r01,r11)
			// y_angle = atan2(-r20,r22)
			res.xAngle = asin((*ptr)(2, 1));
			res.zAngle = atan2(-(*ptr)(0, 1), (*ptr)(1, 1));
			res.yAngle = atan2(-(*ptr)(2, 0), (*ptr)(2, 2));
			res.unique = 1;
		}
		else
		{
			// x_angle = -pi/2
			// y_angle - z_angle = atan2(r02,r00)
			// WARNING.  The solution is not unique.  Choosing y_angle = 0.
			res.xAngle = -static_cast<float>(M_PI_2);
			res.zAngle = -atan2((*ptr)(0, 2), (*ptr)(0, 0));
			res.yAngle = 0;
			res.unique = 0;
		}
	}
	else
	{
		// x_angle = +pi/2
		// y_angle + z_angle = atan2(r02,r00)
		// WARNING.  The solution is not unique.  Choosing y_angle = 0.
		res.xAngle = static_cast<float>(M_PI_2);
		res.zAngle = atan2((*ptr)(0, 2), (*ptr)(0, 0));
		res.yAngle = 0;
		res.unique = 0;
	}
	return res;
}

template<class RealType, class ResultType, std::enable_if_t<std::is_floating_point<RealType>::value, bool> = true>
ResultType Matrix3ExtractEulerZYX(const Eigen::Matrix<RealType, 3, 3> * ptr)
{
	// +-           -+   +-                                      -+
	// | r00 r01 r02 |   |  cy*cz  cz*sx*sy-cx*sz  cx*cz*sy+sx*sz |
	// | r10 r11 r12 | = |  cy*sz  cx*cz+sx*sy*sz -cz*sx+cx*sy*sz |
	// | r20 r21 r22 |   | -sy     cy*sx           cx*cy          |
	// +-           -+   +-                                      -+
	ResultType res;
	if ((*ptr)(2, 0) < 1)
	{
		if ((*ptr)(2, 0) > -1)
		{
			// y_angle = asin(-r20)
			// z_angle = atan2(r10,r00)
			// x_angle = atan2(r21,r22)
			res.yAngle = asin(-(*ptr)(2, 0));
			res.zAngle = atan2((*ptr)(1, 0), (*ptr)(0, 0));
			res.xAngle = atan2((*ptr)(2, 1), (*ptr)(2, 2));
			res.unique = 1;
		}
		else
		{
			// y_angle = +pi/2
			// x_angle - z_angle = atan2(r01,r02)
			// WARNING.  The solution is not unique.  Choosing x_angle = 0.
			res.yAngle = static_cast<float>(M_PI_2);
			res.zAngle = -atan2((*ptr)(0, 1), (*ptr)(0, 2));
			res.xAngle = 0;
			res.unique = 0;
		}
	}
	else
	{
		// y_angle = -pi/2
		// x_angle + z_angle = atan2(-r01,-r02)
		// WARNING.  The solution is not unique.  Choosing x_angle = 0;
		res.yAngle = -static_cast<float>(M_PI_2);
		res.zAngle = atan2(-(*ptr)(0, 1), -(*ptr)(0, 2));
		res.xAngle = 0;
		res.unique = 0;
	}
	return res;
}

template<class RealType, std::enable_if_t<std::is_floating_point<RealType>::value, bool> = true>
void EulerFactorXYZ(const Eigen::Matrix<RealType, 3, 3>& m, RealType & a1, RealType & a2, RealType & a3)
{
	EulerResult<RealType> res = Matrix3ExtractEulerXYZ<RealType, EulerResult<RealType> >(&m);
	a1 = res.xAngle;
	a2 = res.yAngle;
	a3 = res.zAngle;
}

template<class RealType, std::enable_if_t<std::is_floating_point<RealType>::value, bool> = true>
void EulerFactorXZY(const Eigen::Matrix<RealType, 3, 3>& m, RealType& a1, RealType& a2, RealType& a3)
{
	EulerResult<RealType> res = Matrix3ExtractEulerXZY<RealType, EulerResult<RealType> >(&m);
	a1 = res.xAngle;
	a2 = res.zAngle;
	a3 = res.yAngle;
}

template<class RealType, std::enable_if_t<std::is_floating_point<RealType>::value, bool> = true>
void EulerFactorYXZ(const Eigen::Matrix<RealType, 3, 3>& m, RealType& a1, RealType& a2, RealType& a3)
{
	EulerResult<RealType> res = Matrix3ExtractEulerYXZ<RealType, EulerResult<RealType> >(&m);
	a1 = res.yAngle;
	a2 = res.xAngle;
	a3 = res.zAngle;
}

template<class RealType, std::enable_if_t<std::is_floating_point<RealType>::value, bool> = true>
void EulerFactorYZX(const Eigen::Matrix<RealType, 3, 3>& m, RealType& a1, RealType& a2, RealType& a3)
{
	EulerResult<RealType> res = Matrix3ExtractEulerYZX<RealType, EulerResult<RealType> >(&m);
	a1 = res.yAngle;
	a2 = res.zAngle;
	a3 = res.xAngle;
}

template<class RealType, std::enable_if_t<std::is_floating_point<RealType>::value, bool> = true>
void EulerFactorZXY(const Eigen::Matrix<RealType, 3, 3>& m, RealType& a1, RealType& a2, RealType& a3)
{
	EulerResult<RealType> res = Matrix3ExtractEulerZXY<RealType, EulerResult<RealType> >(&m);
	a1 = res.zAngle;
	a2 = res.xAngle;
	a3 = res.yAngle;
}

template<class RealType, std::enable_if_t<std::is_floating_point<RealType>::value, bool> = true>
void EulerFactorZYX(const Eigen::Matrix<RealType, 3, 3>& m, RealType& a1, RealType& a2, RealType& a3)
{
	EulerResult<RealType> res = Matrix3ExtractEulerZYX<RealType, EulerResult<RealType> >(&m);
	a1 = res.zAngle;
	a2 = res.yAngle;
	a3 = res.xAngle;
}
