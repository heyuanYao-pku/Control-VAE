/*************************************************************************
*                                                                       *
* Open Dynamics Engine, Copyright (C) 2001,2002 Russell L. Smith.       *
* All rights reserved.  Email: russ@q12.org   Web: www.q12.org          *
*                                                                       *
* This library is free software; you can redistribute it and/or         *
* modify it under the terms of EITHER:                                  *
*   (1) The GNU Lesser General Public License as published by the Free  *
*       Software Foundation; either version 2.1 of the License, or (at  *
*       your option) any later version. The text of the GNU Lesser      *
*       General Public License is included with this library in the     *
*       file LICENSE.TXT.                                               *
*   (2) The BSD-style license that is included with this library in     *
*       the file LICENSE-BSD.TXT.                                       *
*                                                                       *
* This library is distributed in the hope that it will be useful,       *
* but WITHOUT ANY WARRANTY; without even the implied warranty of        *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the files    *
* LICENSE.TXT and LICENSE-BSD.TXT for more details.                     *
*                                                                       *
*************************************************************************/
// Add by Zhenhua Song
#pragma once
#include <ode/common.h>
// #include <util.h>
#include "joints/joint.h"

typedef struct dArrayWithShapeStruct
{
    dReal* data;
    unsigned int row;
    unsigned int column;
    unsigned int skip4;
    unsigned int row_skip;
}dArrayWithShape;

typedef dArrayWithShape* dArrayWithShapePtr;
void dArrayWithShapeReset(dArrayWithShapePtr arr);
void dArrayWithShapeClear(dArrayWithShapePtr arr);

typedef struct dIntArrayWithShapeStruct
{
    int * data;
    unsigned int row;
    unsigned int column;
    unsigned int skip4;
    unsigned int row_skip;
}dIntArrayWithShape;

typedef dIntArrayWithShape* dIntArrayWithShapePtr;
void dIntArrayWithShapeReset(dIntArrayWithShapePtr arr);

typedef struct dJointArrayWithShapeStruct
{
    dJointID * data;
    unsigned int* m;
    unsigned int* nub;
    unsigned int cnt;
}dJointArrayWithShape;

typedef dJointArrayWithShape* dJointArrayWithShapePtr;
void dJointArrayWithShapeReset(dJointArrayWithShapePtr arr);

//////////////////////////////////////////////////////
struct WorldStepFeatureInfo // Add by Zhenhua Song
{
    dArrayWithShape lcp_lambda;
    dArrayWithShape lcp_w;
    dArrayWithShape lcp_lo;
    dArrayWithShape lcp_hi;

    dArrayWithShape lcp_a;
    dArrayWithShape lcp_rhs;

    dArrayWithShape jacobian;
    dArrayWithShape joint_c; // Jv = c

    dArrayWithShape j_minv_j;
    dArrayWithShape cfm;

    dIntArrayWithShape findex;
    
    dArrayWithShape damping; // damping is cala in dampedstep
    dJointArrayWithShape joints;


};

typedef WorldStepFeatureInfo * WorldStepFeatureInfoPtr;

void WorldStepFeatureInfoReset(WorldStepFeatureInfoPtr info);
void WorldStepFeatureInfoClear(WorldStepFeatureInfoPtr info);

struct dJointWithInfo1
{
    dxJoint* joint;
    dxJoint::Info1 info;
};

ODE_API int dPADFunction(int n);

ODE_API void PrintMat(const char *str, dReal *m, unsigned int rows, unsigned int cols, int skip4 /* bool */, unsigned int rowskip, int saveToFile = 1 /* bool */);

ODE_API void Multiply2_p8r(dReal *A, const dReal *B, const dReal *C, unsigned int p, unsigned int r, int Askip);

ODE_API void MultiplyAdd2_p8r (dReal *A, const dReal *B, const dReal *C, unsigned int p, unsigned int r, unsigned int Askip);

ODE_API void MultiplySub0_p81 (dReal *A, const dReal *B, const dReal *C, unsigned int p);

ODE_API void MultiplyAdd1_8q1 (dReal *A, const dReal *B, const dReal *C, unsigned int q);

ODE_API void Multiply1_8q1 (dReal *A, const dReal *B, const dReal *C, unsigned int q);

ODE_API void Multiply2_p4r (dReal *A, const dReal *B, const dReal *C, unsigned int p, unsigned int r, int Askip);

ODE_API void MultiplyAdd2_p4r(dReal *A, const dReal *B, const dReal *C,
    unsigned int p, unsigned int r, unsigned int Askip,
    unsigned int Bskip = 4, unsigned int Cskip = 8);

ODE_API void MultiplyAdd1_4q1 (dReal *A, const dReal *B, const dReal *C, unsigned int q);

// Add by Zhenhua Song
ODE_API void ODEMat3ToDenseMat3(const dMatrix3 odeMat3, dReal * denseMat3Out, int offset);

// Add by Zhenhua Song
ODE_API void DenseMat3ToODEMat3(dMatrix3 odeMat3, const dReal * denseMat3In, int offset);
