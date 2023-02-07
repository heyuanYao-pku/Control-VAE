#pragma once
#include <Eigen/Dense>
#include <vector>
#include <iostream>

typedef Eigen::Vector3d Vector3d;
typedef Eigen::Quaterniond Quaterniond;
typedef Eigen::Matrix3d Matrix3d;
typedef Eigen::MatrixXd MatrixXd;
typedef Eigen::VectorXd VectorXd;
typedef Eigen::ArrayXd ArrayXd;
typedef Eigen::ArrayXXd ArrayXXd;

typedef char CStr4[4];
typedef double CVec3[3];
typedef double CQuat4[4];
typedef CVec3 CMat33[3];

#define dCASSERT(a) {if (!(a)) { printf("assertion \"" #a "\" failed in %s:%u", __FILE__, __LINE__); std::exit(1); }}

// C pointer wrapper
void EigenVec3ToCVec3(const Vector3d& a, CVec3 b);
void EigenQuat4ToCQuat4(const Quaterniond& a, CQuat4 b);
void EigenVec3ArrToCVec3Arr(const std::vector<Vector3d>& a, CVec3* b);
void EigenMat33ToCMat33(const Matrix3d& a, CMat33 b);
void EigenMat33ArrToCMat33Arr(const std::vector<Matrix3d>& a, CMat33* b);
void CVec3ArrToEigenVec3Arr(const CVec3* a, size_t aSize, std::vector<Vector3d>& b);


// Wrapper of std::vector<Eigen::Vector3d>
class std_vector_Vector3d {
public:
	std::vector<Vector3d> data;
	std_vector_Vector3d() = default;
	std_vector_Vector3d(size_t Size) { resize(Size); }
	size_t size() const { return data.size(); }
	void resize(size_t Size) { data.resize(Size, Vector3d::Zero()); }
	double getValue(size_t i, size_t j) const { return data[i](j); }
	void setValue(size_t i, size_t j, double value) { data[i](j) = value; }
	void setValue(size_t i, double x, double y, double z) { data[i](0) = x; data[i](1) = y; data[i](2) = z; }
};

// Wrapper of const std::vector<Eigen::Vector3d> *
class std_vector_Vector3d_ptr {
public:
	const std::vector<Vector3d>* ptr;
	std_vector_Vector3d_ptr() { this->ptr = nullptr; };
	std_vector_Vector3d_ptr(const std::vector<Vector3d>* ptr_) { this->ptr = ptr_; }
	size_t size() const { dCASSERT(ptr != nullptr); return ptr->size(); }
	int is_null() const { return this->ptr == nullptr; }
	double getValue(size_t i, size_t j) const { return (*ptr)[i](j); }
};

// Wrapper of std::vector<Eigen::Quaterniond> data;
class std_vector_Quaterniond {
public:
	std::vector<Quaterniond> data;
	size_t size() const { return data.size(); }
	void resize(size_t Size) { data.resize(Size, Quaterniond::Identity()); }
	double getX(size_t i) const { return data[i].x(); }
	double getY(size_t i) const { return data[i].y(); }
	double getZ(size_t i) const { return data[i].z(); }
	double getW(size_t i) const { return data[i].w(); }
	void setValue(size_t i, double x, double y, double z, double w) { data[i].x() = x; data[i].y() = y; data[i].z() = z; data[i].w() = w; }
};

// Wrapper of std::vector<std::vector<Eigen::Quaterniond>> data
class std_vector_std_vector_Quaterniond {
public:
	std::vector<std::vector<Quaterniond>> data;
	size_t size_0() const { return data.size(); }
	size_t size_1() const { return data.size() == 0 ? 0 : data[0].size(); }
	void resize(size_t size_0, size_t size_1) 
	{ 
		data.resize(size_0);
		for (size_t i = 0; i < size_0; i++)
		{
			data[i].resize(size_1, Quaterniond::Identity());
		}
	}
	double getX(size_t size_0, size_t size_1) const { return data[size_0][size_1].x(); }
	double getY(size_t size_0, size_t size_1) const { return data[size_0][size_1].y(); }
	double getZ(size_t size_0, size_t size_1) const { return data[size_0][size_1].z(); }
	double getW(size_t size_0, size_t size_1) const { return data[size_0][size_1].w(); }
	void setValue(size_t size_0, size_t size_1, double x, double y, double z, double w)
	{
		Quaterniond& q = data[size_0][size_1];
		q.x() = x; q.y() = y; q.z() = z; q.w() = w;
	}
};

// Wrapper of const std::vector<Eigen::Quaterniond>* ptr;
class std_vector_Quaterniond_ptr 
{
public:
	const std::vector<Quaterniond>* ptr;
	std_vector_Quaterniond_ptr() { this->ptr = nullptr; };
	std_vector_Quaterniond_ptr(const std::vector<Quaterniond>* ptr_) { this->ptr = ptr_; }
	size_t size() const { dCASSERT(ptr != nullptr); return ptr->size(); }
	int is_null() const { return this->ptr == nullptr; }
	double getX(size_t i) const { return (*ptr)[i].x(); }
	double getY(size_t i) const { return (*ptr)[i].y(); }
	double getZ(size_t i) const { return (*ptr)[i].z(); }
	double getW(size_t i) const { return (*ptr)[i].w(); }
};

// Wrapper of std::vector<Eigen::Matrix3d>
class std_vector_Matrix3d {
public:
	std::vector<Matrix3d> data;
	size_t size() const { return data.size(); }
	void resize(size_t Size) { data.resize(Size, Matrix3d::Zero()); }
	double getValue(size_t i, size_t j, size_t k) const
	{
		const Matrix3d& m = data[i];
		double res = m(j, k);
		return res;
	}
	void setValue(size_t i, size_t j, size_t k, double value) { data[i](j, k) = value; }
	void setValue(size_t i, double a00, double a01, double a02, double a10, double a11, double a12, double a20, double a21, double a22) 
	{
		data[i] << a00, a01, a02, a10, a11, a12, a20, a21, a22;
	}
};

// Wrapper of const std::vector<Eigen::Matrix3d>* ptr
class std_vector_Matrix3d_ptr {
public:
	const std::vector<Matrix3d>* ptr;
	std_vector_Matrix3d_ptr() { ptr = nullptr; }
	std_vector_Matrix3d_ptr(const std::vector<Matrix3d>* ptr_) { this->ptr = ptr; }
	size_t size() const { dCASSERT(ptr != nullptr); return ptr->size(); }
	int is_null() const { return this->ptr == nullptr; }
	double getValue(size_t i, size_t j, size_t k) const { return (*ptr)[i](j, k); }
};

// Wrapper of Eigen::Vector3d
class Eigen_Vector3d {
public:
	Vector3d data;
	Eigen_Vector3d() { data = Vector3d::Zero(); }
	double getValue(size_t i) const { return data(i); }
	void setValue(size_t i, double value) { data[i] = value; }
	void print() const { std::cout << this->data << std::endl; }
};

// Wrapper of Eigen::Quaterniond
class Eigen_Quaterniond {
public:
	Quaterniond data;
	Eigen_Quaterniond() { data = Quaterniond::Identity(); }
	double x() const { return data.x(); }
	double y() const { return data.y(); }
	double z() const { return data.z(); }
	double w() const { return data.w(); }
	void setValue(double x, double y, double z, double w) { data.x() = x; data.y() = y; data.z() = z; data.w() = w; }
	// void print() const { std::cout << data << std::endl; }
};

// Wrapper of Eigen::Matrix3d
class Eigen_Matrix3d {
public:
	Matrix3d data;
	Eigen_Matrix3d() { data = Matrix3d::Zero(); }
	double getValue(size_t i, size_t j) const { return data(i, j); }
	void setValue(size_t i, size_t j, double value) { data(i, j) = value; }
	void PrintData() const { std::cout << this->data << std::endl; }
};

// Wrapper of Eigen::MatrixXd
class Eigen_MatrixXd {
public:
	MatrixXd data;
	Eigen_MatrixXd() {}
	Eigen_MatrixXd(size_t rows, size_t cols) { this->resize(rows, cols); }
	double getValue(size_t i, size_t j) const { return data(i, j); }
	void setValue(size_t i, size_t j, double value) { data(i, j) = value; }
	void resize(size_t rows, size_t cols) { this->data = MatrixXd(rows, cols); }
	size_t rows() const { return this->data.rows(); }
	size_t cols() const { return this->data.cols(); }
	void PrintData() const { std::cout << this->data << std::endl; }
};

// Wrapper of Eigen::VectorXd
class Eigen_VectorXd {
public:
	VectorXd data;
	Eigen_VectorXd() { }
	Eigen_VectorXd(size_t i) { resize(i); }
	size_t size() const { return data.size(); }
	void resize(size_t i) { this->data = VectorXd(i); }
	double getValue(size_t i) const { return this->data(i); }
	void setValue(size_t i, double value) { this->data(i) = value; }
	void PrintData() const { std::cout << this->data << std::endl; }
};

// Wrapper of Eigen::ArrayXd
class Eigen_ArrayXd {
public:
	ArrayXd data;
	Eigen_ArrayXd() {}
	Eigen_ArrayXd(size_t Size) { data = ArrayXd(Size); }
	double getValue(size_t i) const { return data(i); }
	void setValue(size_t i, double value) { data(i) = value; }
	void PrintData() const { std::cout << this->data << std::endl; }
};

// Wrapper of Eigen::ArrayXXd
class Eigen_ArrayXXd {
public:
	ArrayXXd data;
	Eigen_ArrayXXd() {}
	double getValue(size_t i, size_t j) const { return data(i, j); }
	void setValue(size_t i, size_t j, double value) { data(i, j) = value; }
	void PrintData() const { std::cout << this->data << std::endl; }
};

