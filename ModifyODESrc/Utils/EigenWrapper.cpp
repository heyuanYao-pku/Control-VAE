# include <EigenExtension/EigenBindingWrapper.h>

void EigenVec3ToCVec3(const Vector3d& a, CVec3 b)
{
    b[0] = a.x();
    b[1] = a.y();
    b[2] = a.z();
}

void EigenQuat4ToCQuat4(const Quaterniond& a, CQuat4 b)
{
    b[0] = a.x();
    b[1] = a.y();
    b[2] = a.z();
    b[3] = a.w();
}

void EigenVec3ArrToCVec3Arr(const std::vector<Vector3d>& a, CVec3* b)
{
    dCASSERT(b != NULL);
    size_t tot = a.size();
    for (size_t i = 0; i < tot; i++)
    {
        EigenVec3ToCVec3(a[i], b[i]);
    }
}

void EigenMat33ToCMat33(const Matrix3d& a, CMat33 b)
{
    b[0][0] = a(0, 0); b[0][1] = a(0, 1); b[0][2] = a(0, 2);
    b[1][0] = a(1, 0); b[1][1] = a(1, 1); b[1][2] = a(1, 2);
    b[2][0] = a(2, 0); b[2][1] = a(2, 1); b[2][2] = a(2, 2);
}

void EigenMat33ArrToCMat33Arr(const std::vector<Matrix3d>& a, CMat33* b)
{
    dCASSERT(b != NULL);
    size_t tot = a.size();
    for (size_t i = 0; i < tot; i++)
    {
        EigenMat33ToCMat33(a[i], b[i]);
    }
}

void CVec3ArrToEigenVec3Arr(const CVec3* a, size_t aSize, std::vector<Vector3d>& b)
{
    dCASSERT(a != NULL);
    b.resize(aSize, Vector3d::Zero());
    for (size_t i = 0; i < aSize; i++)
    {
        b[i] << a[i][0], a[i][1], a[i][2];
    }
}
