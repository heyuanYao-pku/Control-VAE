#include <ode/extutils.h>

void dArrayWithShapeReset(dArrayWithShapePtr arr)
{
    arr->data = NULL;
    arr->row = 0;
    arr->column = 0;
    arr->skip4 = 0;
    arr->row_skip = 0;
}

void dArrayWithShapeClear(dArrayWithShapePtr arr)
{
    if (arr->data != NULL)
    {
        free(arr->data);
    }
    dArrayWithShapeReset(arr);
}

void dIntArrayWithShapeReset(dIntArrayWithShapePtr arr)
{
    arr->data = NULL;
    arr->row = 0;
    arr->column = 0;
    arr->skip4 = 0;
    arr->row_skip = 0;
}

void dIntArrayWithShapeClear(dIntArrayWithShapePtr arr)
{
    if (arr->data != NULL)
    {
        free(arr->data);
    }
    dIntArrayWithShapeReset(arr);
}

void dJointArrayWithShapeReset(dJointArrayWithShapePtr arr)
{
    arr->data = NULL;
    arr->m = NULL;
    arr->nub = NULL;

    arr->cnt = 0;
}

void dJointArrayWithShapeClear(dJointArrayWithShapePtr arr)
{
    if (arr->data != NULL)
    {
        free(arr->data);
    }
    if (arr->m != NULL)
    {
        free(arr->m);
    }
    if (arr->nub != NULL)
    {
        free(arr->nub);
    }
    dJointArrayWithShapeReset(arr);
}

void WorldStepFeatureInfoReset(WorldStepFeatureInfoPtr info)
{
    dArrayWithShapeReset(&(info->lcp_lambda));
    dArrayWithShapeReset(&(info->lcp_w));
    dArrayWithShapeReset(&(info->lcp_lo));
    dArrayWithShapeReset(&(info->lcp_hi));

    dArrayWithShapeReset(&(info->lcp_a));
    dArrayWithShapeReset(&(info->lcp_rhs));

    dArrayWithShapeReset(&(info->jacobian));
    dArrayWithShapeReset(&(info->joint_c));
    // printf("joint_c ptr %lld\n", info->joint_c.data);

    dArrayWithShapeReset(&(info->j_minv_j));
    dArrayWithShapeReset(&(info->cfm));
    dIntArrayWithShapeReset(&(info->findex));

    dArrayWithShapeReset(&(info->damping));

    dJointArrayWithShapeReset(&(info->joints));

}

void WorldStepFeatureInfoClear(WorldStepFeatureInfoPtr info)
{
    dArrayWithShapeClear(&(info->lcp_lambda));
    dArrayWithShapeClear(&(info->lcp_w));
    dArrayWithShapeClear(&(info->lcp_lo));
    dArrayWithShapeClear(&(info->lcp_hi));

    dArrayWithShapeClear(&(info->lcp_a));
    dArrayWithShapeClear(&(info->lcp_rhs));

    dArrayWithShapeClear(&(info->jacobian));
    dArrayWithShapeClear(&(info->joint_c));

    dArrayWithShapeClear(&(info->j_minv_j));
    dArrayWithShapeClear(&(info->cfm));

    dIntArrayWithShapeClear(&(info->findex));

    dArrayWithShapeClear(&(info->damping));

    dJointArrayWithShapeClear(&(info->joints));
}

int dPADFunction(int n)
{
    return dPAD(n);
}

//////////////////////////////////////////////////////////////////////
ODE_API void PrintMat(
    const char *str, // save filename
    dReal *m, // data
    unsigned int rows, //  
    unsigned int cols, 
    int skip4 /* bool */,
    unsigned int rowskip,
    int saveToFile /* bool */)
{
    printf("---- %s ----\n", str);
    if (cols == 0 || rows == 0)
        return;

    dReal *mlocal = m;
    for (unsigned int i = 0; i < rows; ++i, mlocal += rowskip)
    {
        for (unsigned int j = 0; j < cols; ++j)
        {
            if (skip4 && j % 4 == 3)
                continue;

            printf(" %12.9f", mlocal[j]);
        }

        printf("\n");
    }

    if (saveToFile)
    {
        FILE *f = fopen(str, "w"); // Modified by Zhenhua Song. Not Use fopen_s
        mlocal = m;
        for (unsigned int i = 0; i < rows; ++i, mlocal += rowskip)
        {
            for (unsigned int j = 0; j < cols; ++j)
            {
                if (skip4 && j % 4 == 3)
                    continue;

                fprintf(f, " %12.9f", mlocal[j]);
            }

            fprintf(f, "\n");
        }
        fclose(f);
    }
}

//****************************************************************************
// special matrix multipliers

// this assumes the 4th and 8th rows of B and C are zero.

ODE_API void Multiply2_p8r(dReal *A, const dReal *B, const dReal *C, unsigned int p, unsigned int r, int Askip)
{
    dIASSERT(p > 0 && r > 0 && A && B && C);
    const int Askip_munus_r = Askip - r;
    dReal *aa = A;
    const dReal *bb = B;
    for (unsigned int i = p; i != 0; --i) {
        const dReal *cc = C;
        for (unsigned int j = r; j != 0; --j) {
            dReal sum;
            sum = bb[0] * cc[0];
            sum += bb[1] * cc[1];
            sum += bb[2] * cc[2];
            sum += bb[4] * cc[4];
            sum += bb[5] * cc[5];
            sum += bb[6] * cc[6];
            *(aa++) = sum;
            cc += 8;
        }
        bb += 8;
        aa += Askip_munus_r;
    }
}

// this assumes the 4th and 8th rows of B and C are zero.

ODE_API void MultiplyAdd2_p8r (dReal *A, const dReal *B, const dReal *C, unsigned int p, unsigned int r, unsigned int Askip)
{
  dIASSERT (p>0 && r>0 && A && B && C);
  const unsigned int Askip_munus_r = Askip - r;
  dIASSERT(Askip >= r);
  dReal *aa = A;
  const dReal *bb = B;
  for (unsigned int i = p; i != 0; --i) {
    const dReal *cc = C;
    for (unsigned int j = r; j != 0; --j) {
      dReal sum;
      sum  = bb[0]*cc[0];
      sum += bb[1]*cc[1];
      sum += bb[2]*cc[2];
      sum += bb[4]*cc[4];
      sum += bb[5]*cc[5];
      sum += bb[6]*cc[6];
      *(aa++) += sum; 
      cc += 8;
    }
    bb += 8;
    aa += Askip_munus_r;
  }
}


// this assumes the 4th and 8th rows of B are zero.

ODE_API void MultiplySub0_p81 (dReal *A, const dReal *B, const dReal *C, unsigned int p)
{
  dIASSERT (p>0 && A && B && C);
  dReal *aa = A;
  const dReal *bb = B;
  for (unsigned int i = p; i != 0; --i) {
    dReal sum;
    sum  = bb[0]*C[0];
    sum += bb[1]*C[1];
    sum += bb[2]*C[2];
    sum += bb[4]*C[4];
    sum += bb[5]*C[5];
    sum += bb[6]*C[6];
    *(aa++) -= sum;
    bb += 8;
  }
}


// this assumes the 4th and 8th rows of B are zero.

ODE_API void MultiplyAdd1_8q1 (dReal *A, const dReal *B, const dReal *C, unsigned int q)
{
  dIASSERT (q>0 && A && B && C);
  const dReal *bb = B;
  dReal sum0 = 0, sum1 = 0, sum2 = 0, sum4=0, sum5 = 0, sum6 = 0;
  for (unsigned int k = 0; k < q; ++k) {
    const dReal C_k = C[k];
    sum0 += bb[0] * C_k;
    sum1 += bb[1] * C_k;
    sum2 += bb[2] * C_k;
    sum4 += bb[4] * C_k;
    sum5 += bb[5] * C_k;
    sum6 += bb[6] * C_k;
    bb += 8;
  }
  A[0] += sum0;
  A[1] += sum1;
  A[2] += sum2;
  A[4] += sum4;
  A[5] += sum5;
  A[6] += sum6;
}


// this assumes the 4th and 8th rows of B are zero.

ODE_API void Multiply1_8q1 (dReal *A, const dReal *B, const dReal *C, unsigned int q)
{
  const dReal *bb = B;
  dReal sum0 = 0, sum1 = 0, sum2 = 0, sum4=0, sum5 = 0, sum6 = 0;
  for (unsigned int k = 0; k < q; ++k) {
    const dReal C_k = C[k];
    sum0 += bb[0] * C_k;
    sum1 += bb[1] * C_k;
    sum2 += bb[2] * C_k;
    sum4 += bb[4] * C_k;
    sum5 += bb[5] * C_k;
    sum6 += bb[6] * C_k;
    bb += 8;
  }
  A[0] = sum0;
  A[1] = sum1;
  A[2] = sum2;
  A[4] = sum4;
  A[5] = sum5;
  A[6] = sum6;
}

ODE_API void Multiply2_p4r (dReal *A, const dReal *B, const dReal *C, unsigned int p, unsigned int r, int Askip)
{
    dIASSERT(p>0 && r>0 && A && B && C);
    const int Askip_munus_r = Askip - r;
    dReal *aa = A;
    const dReal *bb = B;
    for (unsigned int i = p; i != 0; --i) {
        const dReal *cc = C;
        for (unsigned int j = r; j != 0; --j) {
            dReal sum;
            sum = bb[0] * cc[0];
            sum += bb[1] * cc[1];
            sum += bb[2] * cc[2];
            *(aa++) = sum;
            cc += 8;
        }
        bb += 4;
        aa += Askip_munus_r;
    }
}

// A+=BC'
ODE_API void MultiplyAdd2_p4r(dReal *A, const dReal *B, const dReal *C,
    unsigned int p, unsigned int r, unsigned int Askip,
    unsigned int Bskip, unsigned int Cskip)
{
    dIASSERT(p>0 && r>0 && A && B && C);
    const unsigned int Askip_munus_r = Askip - r;
    dIASSERT(Askip >= r);
    dReal *aa = A;
    const dReal *bb = B;
    for (unsigned int i = p; i != 0; --i) {
        const dReal *cc = C;
        for (unsigned int j = r; j != 0; --j) {
            dReal sum;
            sum = bb[0] * cc[0];
            sum += bb[1] * cc[1];
            sum += bb[2] * cc[2];
            *(aa++) += sum;
            cc += Cskip;//8;
        }
        bb += Bskip;//4;
        aa += Askip_munus_r;
    }
}

// A=BC
ODE_API void MultiplyAdd1_4q1 (dReal *A, const dReal *B, const dReal *C, unsigned int q)
{
  dIASSERT (q>0 && A && B && C);
  const dReal *bb = B;
  dReal sum0 = 0, sum1 = 0, sum2 = 0;
  for (unsigned int k = 0; k < q; ++k) {
    const dReal C_k = C[k];
    sum0 += bb[0] * C_k;
    sum1 += bb[1] * C_k;
    sum2 += bb[2] * C_k;
    bb += 4;
  }
  A[0] += sum0;
  A[1] += sum1;
  A[2] += sum2;
}

ODE_API void ODEMat3ToDenseMat3(const dMatrix3 odeMat3, dReal * denseMat3Out, int offset)
{
  dReal * denseMat3 = denseMat3Out + offset;

  denseMat3[0] = odeMat3[0];
  denseMat3[1] = odeMat3[1];
  denseMat3[2] = odeMat3[2];
  denseMat3[3] = odeMat3[4];
  denseMat3[4] = odeMat3[5];
  denseMat3[5] = odeMat3[6];
  denseMat3[6] = odeMat3[8];
  denseMat3[7] = odeMat3[9];
  denseMat3[8] = odeMat3[10];
}

ODE_API void DenseMat3ToODEMat3(dMatrix3 odeMat3, const dReal * denseMat3In, int offset)
{
  const dReal * denseMat3 = denseMat3In + offset;

  odeMat3[0] = denseMat3[0];
  odeMat3[1] = denseMat3[1];
  odeMat3[2] = denseMat3[2];
  odeMat3[3] = 0;
  odeMat3[4] = denseMat3[3];
  odeMat3[5] = denseMat3[4];
  odeMat3[6] = denseMat3[5];
  odeMat3[7] = 0;
  odeMat3[8] = denseMat3[6];
  odeMat3[9] = denseMat3[7];
  odeMat3[10] = denseMat3[8];
  odeMat3[11] = 0;
}