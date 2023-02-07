#pragma once
#include <Eigen/Dense>

void TransInertia(Eigen::Matrix3d& I, double mass, double tx, double ty, double tz);
