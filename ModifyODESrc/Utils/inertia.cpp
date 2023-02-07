#include <inertia.h>

void TransInertia(Eigen::Matrix3d& I, double mass, double tx, double ty, double tz)
{
	Eigen::Matrix3d t = Eigen::Matrix3d::Zero();
	t(0, 0) = mass * (ty * ty + tz * tz); t(0, 1) = -mass * tx * ty;            t(0, 2) = -mass * tx * tz;
	t(1, 0) = t(0, 1);                    t(1, 1) = mass * (tx * tx + tz * tz); t(1, 2) = -mass * ty * tz;
	t(2, 0) = t(0, 2);                    t(2, 1) = t(1, 2);                    t(2, 2) = mass * (tx * tx + ty * ty);

	I += t;
}