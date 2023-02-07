# cython: language_level=3

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset

cdef extern from "EigenExtension/EigenBindingWrapper.h" nogil:
    # Wrapper of std::vector<Eigen::Vector3d>
    cdef cppclass std_vector_Vector3d:
        std_vector_Matrix3d()
        std_vector_Matrix3d(size_t Size)
        size_t size() const
        void resize(size_t Size)
        double getValue(size_t i, size_t j) const
        void setValue(size_t i, size_t j, double value)
        void setValue(size_t i, double x, double y, double z)

    # Wrapper of const std::vector<Eigen::Vector3d> *
    cdef cppclass std_vector_Vector3d_ptr:
        size_t size() const
        int is_null() const
        double getValue(size_t i, size_t j) const

    # Wrapper of std::vector<Eigen::Quaterniond>
    cdef cppclass std_vector_Quaterniond:
        size_t size() const
        void resize(size_t Size)
        double getX(size_t i) const
        double getY(size_t i) const
        double getZ(size_t i) const
        double getW(size_t i) const
        void setValue(size_t i, double x, double y, double z, double w)

    # Wrapper of std::vector<std::vector<Eigen::Quaterniond>>
    cdef cppclass std_vector_std_vector_Quaterniond:
        size_t size_0() const
        size_t size_1() const
        void resize(size_t size_0, size_t size_1)
        double getX(size_t size_0, size_t size_1) const
        double getY(size_t size_0, size_t size_1) const
        double getZ(size_t size_0, size_t size_1) const
        double getW(size_t size_0, size_t size_1) const
        void setValue(size_t size_0, size_t size_1, double x, double y, double z, double w)

    # Wrapper of 
    cdef cppclass std_vector_Quaterniond_ptr:
        size_t size() const
        int is_null() const
        double getX(size_t i) const
        double getY(size_t i) const
        double getZ(size_t i) const
        double getW(size_t i) const

    # Wrapper of std::vector<Eigen::Matrix3d>
    cdef cppclass std_vector_Matrix3d:
        size_t size() const 
        void resize(size_t Size)
        double getValue(size_t i, size_t j, size_t k) const 
        void setValue(size_t i, size_t j, size_t k, double value)
        void setValue(size_t i, double a00, double a01, double a02, double a10, double a11, double a12, double a20, double a21, double a22) 

    # Wrapper of const std::vector<Eigen::Matrix3d> *
    cdef cppclass std_vector_Matrix3d_ptr:
        size_t size() const
        int is_null() const
        double getValue(size_t i, size_t j, size_t k) const

    # Wrapper of Eigen::Matrix3d
    cdef cppclass Eigen_Matrix3d:
        Eigen_Matrix3d()
        double getValue(size_t i, size_t j) const
        void setValue(size_t i, size_t j, double value)

    # Wrapper of Eigen::Vector3d
    cdef cppclass Eigen_Vector3d:
        Eigen_Vector3d()
        double getValue(size_t i) const
        void setValue(size_t i, double value)
        void PrintData() const

    # Wrapper of Eigen::Quaterniond
    cdef cppclass Eigen_Quaterniond:
        Eigen_Quaterniond()
        double x() const 
        double y() const 
        double z() const 
        double w() const 
        void setValue(double x, double y, double z, double w)
    
    # Wrapper of Eigen::MatrixXd
    cdef cppclass Eigen_MatrixXd:
        Eigen_MatrixXd()
        Eigen_MatrixXd(size_t rows, size_t cols)
        double getValue(size_t i, size_t j) const
        void setValue(size_t i, size_t j, double value)
        void resize(size_t rows, size_t cols)
        size_t rows() const 
        size_t cols() const 
        void PrintData() const
    
    # Wrapper of Eigen::VectorXd
    cdef cppclass Eigen_VectorXd:
        Eigen_VectorXd()
        Eigen_VectorXd(size_t i)
        size_t size() const
        void resize(size_t i)
        double getValue(size_t i) const
        void setValue(size_t i, double value)
        void PrintData() const 
    
    # Wrapper of Eigen::ArrayXd
    cdef cppclass Eigen_ArrayXd:
        Eigen_ArrayXd()
        Eigen_ArrayXd(size_t Size)
        double getValue(size_t i) const
        void setValue(size_t i, double value)
        void PrintData() const 

    # Wrapper of Eigen::ArrayXXd
    cdef cppclass Eigen_ArrayXXd:
        Eigen_ArrayXXd()
        double getValue(size_t i, size_t j) const
        void setValue(size_t i, size_t j, double value)
        void PrintData() const


cdef extern from "EigenExtension/LinearEquSolve.h" nogil:
    void eigen_solve_conjugate_gradient(
        double * a,
        double * b,
        double * x,
        size_t ndim
    )

    void eigen_solve_llt(
        double * a,
        double * b,
        double * x,
        size_t ndim
    )

    void eigen_solve_colPivHouseholderQr(
        double * a,
        double * b,
        double * x,
        size_t ndim
    )