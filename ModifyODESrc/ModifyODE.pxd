# cython: language_level=3
######################################################################
# Python Open Dynamics Engine Wrapper
# Copyright (C) 2004 PyODE developers (see file AUTHORS)
# All rights reserved.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of EITHER:
#   (1) The GNU Lesser General Public License as published by the Free
#       Software Foundation; either version 2.1 of the License, or (at
#       your option) any later version. The text of the GNU Lesser
#       General Public License is included with this library in the
#       file LICENSE.
#   (2) The BSD-style license that is included with this library in
#       the file LICENSE-BSD.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the files
# LICENSE and LICENSE-BSD for more details.
######################################################################

# Add by Zhenhua Song:
# Note: If you forget to input : in this file, there will be some strange errors when compiling

from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
from libc.string cimport memcpy, memset

cdef extern from "ode/ode.h":

    ctypedef double dReal

    # Modify by Zhenhua Song: remove Dummy structs
    cdef struct dxWorld
    cdef struct dxSpace
    cdef struct dxBody
    cdef struct dxGeom
    cdef struct dxJoint
    cdef struct dxJointGroup
    cdef struct dxTriMeshData
    cdef struct dxHeightfieldData

    # Types
    ctypedef dxWorld* dWorldID
    ctypedef dxSpace* dSpaceID
    ctypedef dxBody* dBodyID
    ctypedef dxGeom* dGeomID
    ctypedef dxJoint* dJointID
    ctypedef dxJointGroup* dJointGroupID
    ctypedef dxTriMeshData* dTriMeshDataID
    ctypedef dxHeightfieldData* dHeightfieldDataID
    ctypedef dReal dVector3[4]
    ctypedef dReal dVector4[4]
    ctypedef dReal dMatrix3[4*3]
    ctypedef dReal dMatrix4[4*4]
    ctypedef dReal dMatrix6[8*6]
    ctypedef dReal dQuaternion[4]

    cdef extern dReal dInfinity
    cdef extern int dAMotorUser
    cdef extern int dAMotorEuler

    ctypedef struct dMass:
        dReal    mass
        dVector4 c
        dMatrix3 I

    ctypedef struct dJointFeedback:
        dVector3 f1
        dVector3 t1
        dVector3 f2
        dVector3 t2

    ctypedef void dNearCallback(void* data, dGeomID o1, dGeomID o2)
    ctypedef dReal dHeightfieldGetHeight( void* p_user_data, int x, int z )

    ctypedef struct dSurfaceParameters:
        int mode
        dReal mu

        dReal mu2
        dReal bounce
        dReal bounce_vel
        dReal soft_erp
        dReal soft_cfm
        dReal motion1,motion2
        dReal slip1, slip2

    ctypedef struct dContactGeom:
        dVector3 pos
        dVector3 normal
        dReal depth
        dGeomID g1, g2

    ctypedef struct dContact:
        dSurfaceParameters surface
        dContactGeom geom
        dVector3 fdir1

    # Add by Zhenhua Song. MAX_SHAPE_LENGTH == 4
    ctypedef struct dArrayWithShape:
        dReal* data
        unsigned int row
        unsigned int column
        unsigned int skip4
        unsigned int row_skip

    ctypedef dArrayWithShape * dArrayWithShapePtr
    void dArrayWithShapeReset(dArrayWithShapePtr arr)

    # Add by Zhenhua Song
    ctypedef struct dIntArrayWithShape:
        int* data
        unsigned int row
        unsigned int column
        unsigned int skip4
        unsigned int row_skip

    ctypedef dIntArrayWithShape * dIntArrayWithShapePtr
    void dIntArrayWithShapeReset(dIntArrayWithShapePtr arr)

    ctypedef struct dJointArrayWithShape:
        dJointID * data
        unsigned int * m
        unsigned int * nub
        unsigned int cnt

    # Add by Zhenhua Song
    ctypedef struct WorldStepFeatureInfo:
        dArrayWithShape lcp_lambda
        dArrayWithShape lcp_w
        dArrayWithShape lcp_lo
        dArrayWithShape lcp_hi

        dArrayWithShape lcp_a
        dArrayWithShape lcp_rhs

        dArrayWithShape jacobian
        dArrayWithShape joint_c

        dArrayWithShape j_minv_j
        dArrayWithShape cfm
        dIntArrayWithShape findex

        dArrayWithShape damping

        dJointArrayWithShape joints

    ctypedef WorldStepFeatureInfo * WorldStepFeatureInfoPtr

    void WorldStepFeatureInfoClear(WorldStepFeatureInfoPtr info)
    void WorldStepFeatureInfoReset(WorldStepFeatureInfoPtr info)
    int dPADFunction(int n)  # Add by Zhenhua Song

    # Add by Zhenhua Song
    cdef enum:
        dMaxUserClasses = 4

    # Add by Zhenhua Song
    cdef enum:
        dSphereClass = 0
        dBoxClass
        dCapsuleClass
        dCylinderClass
        dPlaneClass
        dRayClass
        dConvexClass
        dGeomTransformClass
        dTriMeshClass
        dHeightfieldClass

        dFirstSpaceClass
        dSimpleSpaceClass = dFirstSpaceClass
        dHashSpaceClass
        dSweepAndPruneSpaceClass
        dQuadTreeSpaceClass
        dLastSpaceClass = dQuadTreeSpaceClass

        dFirstUserClass
        dLastUserClass = dFirstUserClass + dMaxUserClasses - 1
        dGeomNumClasses

    ctypedef enum dJointType:
        dJointTypeNone = 0
        dJointTypeBall
        dJointTypeHinge
        dJointTypeSlider
        dJointTypeContact
        dJointTypeUniversal
        dJointTypeHinge2
        dJointTypeFixed
        dJointTypeNull
        dJointTypeAMotor
        dJointTypeLMotor
        dJointTypePlane2D
        dJointTypePR
        dJointTypePU
        dJointTypePiston

        # add by Libin Liu
        dJointTypeContact2

        # Add by Zhenhua Song
        dJointTypeContactMaxForce

        # Add by Zhenhua Song
        dJointTypeEmptyBall

    # Add by Zhenhua Song
    cdef enum:
        dParamLoStop = 0
        dParamHiStop
        dParamVel
        dParamFMax
        dParamFudgeFactor
        dParamBounce
        dParamCFM
        dParamStopERP
        dParamStopCFM
        # parameters for suspension
        dParamSuspensionERP
        dParamSuspensionCFM
        dParamERP
        dParamVelMax

        dParamLoStop1 = 0x000
        dParamHiStop1
        dParamVel1
        dParamFMax1
        dParamFudgeFactor1
        dParamBounce1
        dParamCFM1
        dParamStopERP1
        dParamStopCFM1
        # parameters for suspension
        dParamSuspensionERP1
        dParamSuspensionCFM1
        dParamERP1
        dParamVelMax1

        dParamLoStop2 = 0x100
        dParamHiStop2
        dParamVel2
        dParamFMax2
        dParamFudgeFactor2
        dParamBounce2
        dParamCFM2
        dParamStopERP2
        dParamStopCFM2
        # parameters for suspension
        dParamSuspensionERP2
        dParamSuspensionCFM2
        dParamERP2
        dParamVelMax2

        dParamLoStop3 = 0x200
        dParamHiStop3
        dParamVel3
        dParamFMax3
        dParamFudgeFactor3
        dParamBounce3
        dParamCFM3
        dParamStopERP3
        dParamStopCFM3
        # parameters for suspension
        dParamSuspensionERP3
        dParamSuspensionCFM3
        dParamERP3
        dParamVelMax3

    cdef enum:
        dContactMu2		= 0x001
        dContactFDir1	= 0x002
        dContactBounce	= 0x004
        dContactSoftERP	= 0x008
        dContactSoftCFM	= 0x010
        dContactMotion1	= 0x020
        dContactMotion2	= 0x040
        dContactMotionN	= 0x080
        dContactSlip1	= 0x100
        dContactSlip2	= 0x200

        dContactApprox0	= 0x0000
        dContactApprox1_1	= 0x1000
        dContactApprox1_2	= 0x2000
        dContactApprox1	= 0x3000

    # World
    dWorldID dWorldCreate()
    void dWorldDestroy (dWorldID)

    void dCloseODE()
    void dInitODE()

    void dWorldSetGravity (dWorldID w, dReal x, dReal y, dReal z)
    void dWorldGetGravity (dWorldID w, dVector3 gravity)
    void dWorldSetERP (dWorldID w, dReal erp)
    dReal dWorldGetERP (dWorldID w)
    void dWorldSetCFM (dWorldID w, dReal cfm)
    dReal dWorldGetCFM (dWorldID w)
    void dWorldStep (dWorldID w, dReal stepsize)

    # Add by Zhenhua Song
    int dWorldStepWithInfo(dWorldID w, dReal stepsize, WorldStepFeatureInfoPtr feature_info)

    # Add by Zhenhua Song
    int dWorldResortJoint(dWorldID w, dReal stepsize, WorldStepFeatureInfoPtr feature_info)
    void dWorldQuickStep (dWorldID w, dReal stepsize)
    void dWorldSetQuickStepNumIterations (dWorldID w, int num)
    int dWorldGetQuickStepNumIterations (dWorldID w)
    void dWorldSetContactMaxCorrectingVel (dWorldID w, dReal vel)
    dReal dWorldGetContactMaxCorrectingVel (dWorldID w)
    void dWorldSetContactSurfaceLayer (dWorldID w, dReal depth)
    dReal dWorldGetContactSurfaceLayer (dWorldID w)
    void dWorldSetAutoDisableFlag (dWorldID w, int do_auto_disable)
    int dWorldGetAutoDisableFlag (dWorldID w)
    void dWorldSetAutoDisableLinearThreshold (dWorldID w, dReal linear_threshold)
    dReal dWorldGetAutoDisableLinearThreshold (dWorldID w)
    void dWorldSetAutoDisableAngularThreshold (dWorldID w, dReal angular_threshold)
    dReal dWorldGetAutoDisableAngularThreshold (dWorldID w)
    void dWorldSetAutoDisableSteps (dWorldID w, int steps)
    int dWorldGetAutoDisableSteps (dWorldID w)
    void dWorldSetAutoDisableTime (dWorldID w, dReal time)
    dReal dWorldGetAutoDisableTime (dWorldID w)
    dReal dWorldGetLinearDamping (dWorldID w)
    void dWorldSetLinearDamping (dWorldID w, dReal scale)
    dReal dWorldGetAngularDamping (dWorldID w)
    void dWorldSetAngularDamping (dWorldID w, dReal scale)
    void dWorldImpulseToForce (dWorldID w, dReal stepsize, dReal ix, dReal iy, dReal iz, dVector3 force)

    # Add by Zhenhua Song
    void dWorldDampedStep(dWorldID w, dReal stepsize)

    # Add by Zhenhua Song
    int dWorldDampedStepWithInfo(dWorldID w, dReal stepsize, WorldStepFeatureInfoPtr feature_info)

    # Add by Zhenhua Song
    dJointID dWorldGetFirstJoint(dWorldID)

    # Add by Zhenhua Song
    dJointID dWorldGetNextJoint(dJointID joint)

    # Add by Zhenhua Song
    int dWorldGetNumJoints(dWorldID w)

    # Add by Zhenhua Song
    int dWorldGetNumBallAndHingeJoints(dWorldID w)

    # Add by Zhenhua Song
    dBodyID dWorldGetFirstBody(dWorldID w)

    # Add by Zhenhua Song
    dBodyID dWorldGetNextBody(dBodyID body)

    # Add by Zhenhua Song
    int dWorldGetNumBody(dWorldID w)

    # Body
    dBodyID dBodyCreate (dWorldID w)
    void dBodyDestroy (dBodyID body)

    void  dBodySetData (dBodyID body, void *data)
    void *dBodyGetData (dBodyID body)

    void dBodySetPosition   (dBodyID body, dReal x, dReal y, dReal z)

    # Add by Zhenhua Song
    void dBodySetRotAndQuatNoNorm(dBodyID, dMatrix3 R, dQuaternion q)

    void dBodySetRotation   (dBodyID, dMatrix3 R)
    void dBodySetQuaternion (dBodyID, dQuaternion q)
    void dBodySetLinearVel  (dBodyID, dReal x, dReal y, dReal z)
    void dBodySetAngularVel (dBodyID, dReal x, dReal y, dReal z)
    dReal * dBodyGetPosition   (dBodyID)
    dReal * dBodyGetRotation   (dBodyID)
    dReal * dBodyGetQuaternion (dBodyID)
    dReal * dBodyGetLinearVel  (dBodyID)
    dReal * dBodyGetAngularVel (dBodyID)

    void dBodySetMass (dBodyID, dMass *mass)
    void dBodyGetMass (dBodyID, dMass *mass)

    void dBodyAddForce            (dBodyID, dReal fx, dReal fy, dReal fz)
    void dBodyAddTorque           (dBodyID, dReal fx, dReal fy, dReal fz)
    void dBodyAddRelForce         (dBodyID, dReal fx, dReal fy, dReal fz)
    void dBodyAddRelTorque        (dBodyID, dReal fx, dReal fy, dReal fz)
    void dBodyAddForceAtPos       (dBodyID, dReal fx, dReal fy, dReal fz, dReal px, dReal py, dReal pz)
    void dBodyAddForceAtRelPos    (dBodyID, dReal fx, dReal fy, dReal fz, dReal px, dReal py, dReal pz)
    void dBodyAddRelForceAtPos    (dBodyID, dReal fx, dReal fy, dReal fz, dReal px, dReal py, dReal pz)
    void dBodyAddRelForceAtRelPos (dBodyID, dReal fx, dReal fy, dReal fz, dReal px, dReal py, dReal pz)

    dReal * dBodyGetForce   (dBodyID)
    dReal * dBodyGetTorque  (dBodyID)

    void dBodySetForce(dBodyID, dReal x, dReal y, dReal z)
    void dBodySetTorque(dBodyID, dReal x, dReal y, dReal z)

    void dBodyGetRelPointPos    (dBodyID, dReal px, dReal py, dReal pz, dVector3 result)
    void dBodyGetRelPointVel    (dBodyID, dReal px, dReal py, dReal pz, dVector3 result)
    void dBodyGetPointVel    (dBodyID, dReal px, dReal py, dReal pz, dVector3 result)
    void dBodyGetPosRelPoint (dBodyID, dReal px, dReal py, dReal pz, dVector3 result)
    void dBodyVectorToWorld   (dBodyID, dReal px, dReal py, dReal pz, dVector3 result)
    void dBodyVectorFromWorld (dBodyID, dReal px, dReal py, dReal pz, dVector3 result)

    void dBodySetFiniteRotationMode (dBodyID, int mode)
    void dBodySetFiniteRotationAxis (dBodyID, dReal x, dReal y, dReal z)

    int dBodyGetFiniteRotationMode (dBodyID)
    void dBodyGetFiniteRotationAxis (dBodyID, dVector3 result)

    int dBodyGetNumJoints (dBodyID b)
    dJointID dBodyGetJoint (dBodyID, int index)

    void dBodyEnable (dBodyID)
    void dBodyDisable (dBodyID)
    int dBodyIsEnabled (dBodyID)

    void dBodySetGravityMode (dBodyID b, int mode)
    int dBodyGetGravityMode (dBodyID b)

    void dBodySetDynamic (dBodyID)
    void dBodySetKinematic (dBodyID)
    int dBodyIsKinematic (dBodyID)

    void dBodySetMaxAngularSpeed (dBodyID, dReal max_speed)

    # Add by Zhenhua Song
    dGeomID dBodyGetFirstGeom(dBodyID body)

    # Joints
    dJointID dJointCreateBall (dWorldID w, dJointGroupID)

    dJointID dJointCreateContactMaxForce(dWorldID w, dJointGroupID group, dContact *)
    dJointID dJointCreateEmptyBall(dWorldID w, dJointGroupID group) # Add by Zhenhua Song

    dJointID dJointCreateHinge (dWorldID w, dJointGroupID)
    dJointID dJointCreateSlider (dWorldID w, dJointGroupID)
    dJointID dJointCreateContact (dWorldID w, dJointGroupID, dContact *)
    dJointID dJointCreateUniversal (dWorldID w, dJointGroupID)
    dJointID dJointCreatePR (dWorldID, dJointGroupID)
    dJointID dJointCreateHinge2 (dWorldID, dJointGroupID)
    dJointID dJointCreateFixed (dWorldID, dJointGroupID)
    dJointID dJointCreateNull (dWorldID, dJointGroupID)
    dJointID dJointCreateAMotor (dWorldID, dJointGroupID)
    dJointID dJointCreateLMotor (dWorldID, dJointGroupID)
    dJointID dJointCreatePlane2D (dWorldID, dJointGroupID)

    void dJointDestroy (dJointID)

    # Add by Heyuan Yao
    void dJointEnableImplicitDamping(dJointID)
    # Add by Zhenhua Song
    void dJointDisableImplicitDamping(dJointID)

    void dJointEnable (dJointID)
    void dJointDisable (dJointID)
    int dJointIsEnabled (dJointID)

    dJointGroupID dJointGroupCreate (int max_size)
    void dJointGroupDestroy (dJointGroupID)
    void dJointGroupEmpty (dJointGroupID)

    void dJointAttach (dJointID, dBodyID body1, dBodyID body2)
    void dJointSetData (dJointID, void *data)
    void *dJointGetData (dJointID)
    int dJointGetType (dJointID)
    dBodyID dJointGetBody (dJointID, int index)

    void dJointSetBallAnchor (dJointID, dReal x, dReal y, dReal z)
    void dJointSetHingeAnchor (dJointID, dReal x, dReal y, dReal z)
    void dJointSetHingeAxis (dJointID, dReal x, dReal y, dReal z)
    void dJointSetHingeParam (dJointID, int parameter, dReal value)
    void dJointAddHingeTorque(dJointID joint, dReal torque)
    void dJointSetSliderAxis (dJointID, dReal x, dReal y, dReal z)
    void dJointSetSliderParam (dJointID, int parameter, dReal value)
    void dJointAddSliderForce(dJointID joint, dReal force)
    void dJointSetHinge2Anchor (dJointID, dReal x, dReal y, dReal z)
    void dJointSetHinge2Axis1 (dJointID, dReal x, dReal y, dReal z)
    void dJointSetHinge2Axis2 (dJointID, dReal x, dReal y, dReal z)
    void dJointSetHinge2Param (dJointID, int parameter, dReal value)
    void dJointAddHinge2Torques(dJointID joint, dReal torque1, dReal torque2)
    void dJointSetUniversalAnchor (dJointID, dReal x, dReal y, dReal z)
    void dJointSetUniversalAxis1 (dJointID, dReal x, dReal y, dReal z)
    void dJointSetUniversalAxis2 (dJointID, dReal x, dReal y, dReal z)
    void dJointSetUniversalParam (dJointID, int parameter, dReal value)
    void dJointAddUniversalTorques(dJointID joint, dReal torque1, dReal torque2)
    void dJointSetFixed (dJointID)
    void dJointSetAMotorNumAxes (dJointID, int num)
    void dJointSetAMotorAxis (dJointID, int anum, int rel, dReal x, dReal y, dReal z)
    void dJointSetAMotorAngle (dJointID, int anum, dReal angle)
    void dJointSetAMotorParam (dJointID, int parameter, dReal value)
    void dJointSetAMotorMode (dJointID, int mode)
    void dJointAddAMotorTorques (dJointID, dReal torque1, dReal torque2, dReal torque3)
    void dJointSetLMotorAxis (dJointID, int anum, int rel, dReal x, dReal y, dReal z)
    void dJointSetLMotorNumAxes (dJointID, int num)
    void dJointSetLMotorParam (dJointID, int parameter, dReal value)

    void dJointGetBallAnchor (dJointID j, dVector3 result)
    void dJointGetBallAnchor2 (dJointID j, dVector3 result)

    # Add by Zhenhua Song
    const dReal * dJointGetBallAnchor1Raw(dJointID j)

    # Add by Zhenhua Song
    const dReal * dJointGetBallAnchor2Raw(dJointID j)

    dReal dJointGetBallParam( dJointID j, int parameter )

    void dJointGetHingeAnchor (dJointID j, dVector3 result)
    void dJointGetHingeAnchor2 (dJointID j, dVector3 result)
    void dJointGetHingeAxis (dJointID j, dVector3 result)

    # Add by Zhenhua Song
    void dJointGetHingeAxis1(dJointID j, dVector3 result)

    # Add by Zhenhua Song
    void dJointGetHingeAxis2(dJointID j, dVector3 result)

    dReal dJointGetHingeParam (dJointID j, int parameter)
    dReal dJointGetHingeAngle (dJointID)
    dReal dJointGetHingeAngleRate (dJointID)
    dReal dJointGetSliderPosition (dJointID)
    dReal dJointGetSliderPositionRate (dJointID)
    void dJointGetSliderAxis (dJointID, dVector3 result)
    dReal dJointGetSliderParam (dJointID, int parameter)
    void dJointGetHinge2Anchor (dJointID, dVector3 result)
    void dJointGetHinge2Anchor2 (dJointID, dVector3 result)
    void dJointGetHinge2Axis1 (dJointID, dVector3 result)
    void dJointGetHinge2Axis2 (dJointID, dVector3 result)
    dReal dJointGetHinge2Param (dJointID, int parameter)
    dReal dJointGetHinge2Angle1 (dJointID)
    dReal dJointGetHinge2Angle1Rate (dJointID j)
    dReal dJointGetHinge2Angle2Rate (dJointID j)
    void dJointGetUniversalAnchor (dJointID j, dVector3 result)
    void dJointGetUniversalAnchor2 (dJointID, dVector3 result)
    void dJointGetUniversalAxis1 (dJointID, dVector3 result)
    void dJointGetUniversalAxis2 (dJointID, dVector3 result)
    dReal dJointGetUniversalParam (dJointID, int parameter)
    dReal dJointGetUniversalAngle1 (dJointID)
    dReal dJointGetUniversalAngle2 (dJointID)
    dReal dJointGetUniversalAngle1Rate (dJointID)
    dReal dJointGetUniversalAngle2Rate (dJointID)
    int dJointGetAMotorNumAxes (dJointID)
    void dJointGetAMotorAxis (dJointID, int anum, dVector3 result)
    int dJointGetAMotorAxisRel (dJointID, int anum)
    dReal dJointGetAMotorAngle (dJointID, int anum)
    dReal dJointGetAMotorAngleRate (dJointID, int anum)
    dReal dJointGetAMotorParam (dJointID, int parameter)
    int dJointGetAMotorMode (dJointID)
    int dJointGetLMotorNumAxes (dJointID)
    void dJointGetLMotorAxis (dJointID, int anum, dVector3 result)
    dReal dJointGetLMotorParam (dJointID, int parameter)
    void dJointSetPlane2DXParam (dJointID, int parameter, dReal value)
    void dJointSetPlane2DYParam (dJointID, int parameter, dReal value)
    void dJointSetPlane2DAngleParam (dJointID, int parameter, dReal value)
    dReal dJointGetPRPosition (dJointID j)
    void dJointSetPRAnchor (dJointID j, dReal x, dReal y, dReal z)
    void dJointSetPRAxis1 (dJointID j, dReal x, dReal y, dReal z)
    void dJointSetPRAxis2 (dJointID j, dReal x, dReal y, dReal z)
    void dJointGetPRAnchor (dJointID j, dVector3 result)
    void dJointGetPRAxis1 (dJointID j, dVector3 result)
    void dJointGetPRAxis2 (dJointID j, dVector3 result)

    # Add by Zhenhua Song
    const dReal * dJointGetHingeAnchor1Raw(dJointID)

    # Add by Zhenhua Song
    const dReal * dJointGetHingeAnchor2Raw(dJointID)

    # Add by Zhenhua Song
    unsigned int dJointGetHingeFlags(dJointID j)

    # Add by Zhenhua Song
    void dJointGetHingeAxis1Raw(dJointID j, dVector3 result)

    # Add by Zhenhua Song
    void dJointGetHingeAxis2Raw(dJointID j, dVector3 result)

    # Add by Zhenhua Song
    void dJointGetHingeQRel(dJointID j, dQuaternion q)

    void dJointSetFeedback (dJointID, dJointFeedback *)
    dJointFeedback *dJointGetFeedback (dJointID)

    # Add by Zhenhua Song
    void dJointSetKd_arr(dJointID, const dReal * kd)

    # Add by Zhenhua Song
    void dJointSetKd(dJointID, dReal kdx, dReal kdy, dReal kdz)

    # Add by Zhenhua Song
    const dReal * dJointGetKd(dJointID)

    # Add by Zhenhua Song
    dReal dJointGetContactParam(dJointID j, int parameter)

    # Add by Zhenhua Song
    void dJointSetContactParam(dJointID j, int parameter, dReal value)

    # Add by Zhenhua Song
    void getAnchor(dJointID j, dVector3 result, dVector3 anchor1 )


    int dAreConnected (dBodyID, dBodyID)

    # Mass
    void dMassSetZero (dMass *)
    void dMassSetParameters (dMass *, dReal themass,
            dReal cgx, dReal cgy, dReal cgz,
            dReal I11, dReal I22, dReal I33,
            dReal I12, dReal I13, dReal I23)
    void dMassSetSphere (dMass *, dReal density, dReal radius)
    void dMassSetSphereTotal (dMass *, dReal total_mass, dReal radius)
    void dMassSetCapsule (dMass *, dReal density, int direction, dReal radius, dReal length)
    void dMassSetCapsuleTotal (dMass *, dReal total_mass, int direction, dReal radius, dReal length)
    void dMassSetCylinder (dMass *, dReal density, int direction, dReal radius, dReal length)
    void dMassSetCylinderTotal (dMass *, dReal total_mass, int direction, dReal radius, dReal length)
    void dMassSetBox (dMass *, dReal density, dReal lx, dReal ly, dReal lz)
    void dMassSetBoxTotal (dMass *, dReal total_mass, dReal lx, dReal ly, dReal lz)
    void dMassAdjust (dMass *, dReal newmass)
    void dMassTranslate (dMass *, dReal x, dReal y, dReal z)
    void dMassRotate (dMass *, dMatrix3 R)
    void dMassAdd (dMass *a, dMass *b)
    void dMassSetTrimesh (dMass *, dReal density, dGeomID g)
    void dMassSetTrimeshTotal (dMass *m, dReal total_mass, dGeomID g)

    # Space
#    dSpaceID dSimpleSpaceCreate(int space)
#    dSpaceID dHashSpaceCreate(int space)
    dSpaceID dSimpleSpaceCreate(dSpaceID space)
    dSpaceID dHashSpaceCreate(dSpaceID space)
    dSpaceID dQuadTreeSpaceCreate (dSpaceID space, dVector3 Center, dVector3 Extents, int Depth)

    void dSpaceDestroy (dSpaceID)
    void dSpaceAdd (dSpaceID, dGeomID)
    void dSpaceRemove (dSpaceID, dGeomID)
    int dSpaceQuery (dSpaceID, dGeomID)
    void dSpaceCollide (dSpaceID space, void *data, dNearCallback *callback)
    void dSpaceCollide2 (dGeomID o1, dGeomID o2, void *data, dNearCallback *callback)

    void dHashSpaceSetLevels (dSpaceID space, int minlevel, int maxlevel)
    void dHashSpaceGetLevels (dSpaceID space, int *minlevel, int *maxlevel)

    void dSpaceSetCleanup (dSpaceID space, int mode)
    int dSpaceGetCleanup (dSpaceID space)

    int dSpaceGetNumGeoms (dSpaceID)
    dGeomID dSpaceGetGeom (dSpaceID, int i)

    # Add by Zhenhua Song
    dGeomID dSpaceGetFirstGeom(dSpaceID space)

    # Add by Zhenhua Song
    dGeomID dSpaceGetNextGeom(dGeomID geom)

    # Add by Zhenhua Song
    int dSpaceGetPlaceableCount(dSpaceID space)

    # Add by Zhenhua Song
    int dSpaceGetPlaceableAndPlaneCount(dSpaceID space)

    # Add by Zhenhua Song
    void dSpaceResortGeoms(dSpaceID space);

    # Add by Zhenhua Song
    void * dSpaceGetData(dSpaceID space)

    # Add by Zhenhua Song
    void dSpaceSetData(dSpaceID space, void * data)

    # Geom
    dGeomID dCreateSphere (dSpaceID space, dReal radius)
    dGeomID dCreateBox (dSpaceID space, dReal lx, dReal ly, dReal lz)
    dGeomID dCreatePlane (dSpaceID space, dReal a, dReal b, dReal c, dReal d)
    dGeomID dCreateCapsule (dSpaceID space, dReal radius, dReal length)
    dGeomID dCreateCylinder (dSpaceID space, dReal radius, dReal length)
    dGeomID dCreateGeomGroup (dSpaceID space)

    void dGeomSphereSetRadius (dGeomID sphere, dReal radius)
    void dGeomBoxSetLengths (dGeomID box, dReal lx, dReal ly, dReal lz)
    void dGeomPlaneSetParams (dGeomID plane, dReal a, dReal b, dReal c, dReal d)
    void dGeomCapsuleSetParams (dGeomID ccylinder, dReal radius, dReal length)
    void dGeomCylinderSetParams (dGeomID ccylinder, dReal radius, dReal length)

    dReal dGeomSphereGetRadius (dGeomID sphere)
    void  dGeomBoxGetLengths (dGeomID box, dVector3 result)
    void  dGeomPlaneGetParams (dGeomID plane, dVector4 result)
    void  dGeomCapsuleGetParams (dGeomID ccylinder, dReal *radius, dReal *length)
    void  dGeomCylinderGetParams (dGeomID ccylinder, dReal *radius, dReal *length)

    dReal dGeomSpherePointDepth (dGeomID sphere, dReal x, dReal y, dReal z)
    dReal dGeomBoxPointDepth (dGeomID box, dReal x, dReal y, dReal z)
    dReal dGeomPlanePointDepth (dGeomID plane, dReal x, dReal y, dReal z)
    dReal dGeomCapsulePointDepth (dGeomID ccylinder, dReal x, dReal y, dReal z)

    dGeomID dCreateRay (dSpaceID space, dReal length)
    void dGeomRaySetLength (dGeomID ray, dReal length)
    dReal dGeomRayGetLength (dGeomID ray)
    void dGeomRaySet (dGeomID ray, dReal px, dReal py, dReal pz, dReal dx, dReal dy, dReal dz)
    void dGeomRayGet (dGeomID ray, dVector3 start, dVector3 dir)

    void dGeomSetData (dGeomID, void *)
    void *dGeomGetData (dGeomID)
    void dGeomSetBody (dGeomID, dBodyID)
    dBodyID dGeomGetBody (dGeomID)
    void dGeomSetPosition (dGeomID, dReal x, dReal y, dReal z)
    void dGeomSetRotation (dGeomID, dMatrix3 R)
    void dGeomSetQuaternion (dGeomID, dQuaternion)
    dReal * dGeomGetPosition (dGeomID)
    dReal * dGeomGetRotation (dGeomID)
    void dGeomGetQuaternion (dGeomID, dQuaternion result)
    void dGeomSetOffsetPosition (dGeomID, dReal x, dReal y, dReal z)

    # Add by Zhenhua Song
    void dGeomSetOffsetWorldPosition (dGeomID, dReal x, dReal y, dReal z)

    void dGeomSetOffsetRotation (dGeomID, dMatrix3 R)

    # Add by Zhenhua Song
    void dGeomSetOffsetWorldRotation(dGeomID, dMatrix3 R)

    # Add by Zhenhua Song
    int dGeomIsPlaceable(dGeomID geom)

    # Add by Zhenhua Song
    # void dGeomAppendIgnore(dGeomID now_id, dGeomID other_id)

    # Add by Zhenhua Song
    # int dGeomIsIgnore(dGeomID now_id, dGeomID other_id)

    # Add by Zhenhua Song
    int dGeomGetCharacterID(dGeomID g)

    # Add by Zhenhua Song
    void dGeomSetCharacterID(dGeomID g, int character_id)

    # Add by Zhenhua Song
    int dGeomGetIndex(dGeomID g)

    # Add by Zhenhua Song
    void dGeomSetIndex(dGeomID g, int index)

    # Add by Zhenhua Song, for visualize in Long Ge's draw stuff framework
    void dGeomRenderGetUserColor(dGeomID g, dReal * result)

    # Add by Zhenhua Song, for visualize in Long Ge's draw stuff framework
    void dGeomRenderInUserColor(dGeomID g, const dReal * input_arr)

    # Add by Zhenhua Song, for visualize in Long Ge's draw stuff framework
    void dGeomRenderInDefaultColor(dGeomID g, int value)

    # Add by Zhenhua Song, for visualize in Long Ge's draw stuff framework
    int dGeomIsRenderInDefaultColor(dGeomID g)

    # Add by Zhenhua Song
    void dGeomPlaneGetNearestPointToOrigin(dGeomID g, dVector3 result)

    # Add by Zhenhua Song
    void dGeomPlaneGetQuatFromZAxis(dGeomID g, dQuaternion result)

    # Add by Zhenhua Song
    dGeomID dGeomGetBodyNext (dGeomID)

    # Add by Zhenhua Song
    int dBodyGetNumGeoms(dBodyID)

    # Add by Zhenhua Song
    unsigned int dBodyGetFlags(dBodyID b)

    # Add by Zhenhua Song
    void dBodyGetInertia(dBodyID b, dReal* out)

    # Add by Zhenhua Song
    void dBodyGetInertiaInv(dBodyID b, dReal* out)

    # Add by Zhenhua Song
    dReal* dBodyGetInitInertia(dBodyID b)

    # Add by Zhenhua Song
    dReal* dBodyGetInitInertiaInv(dBodyID b)

    # Add by Zhenhua Song
    dReal dBodyGetMassValue(dBodyID b)

    void dGeomClearOffset (dGeomID)
    dReal * dGeomGetOffsetPosition (dGeomID)
    dReal * dGeomGetOffsetRotation (dGeomID)
    void dGeomDestroy (dGeomID)
    void dGeomGetAABB (dGeomID, dReal aabb[6])
    dReal *dGeomGetSpaceAABB (dGeomID)
    int dGeomIsSpace (dGeomID)
    dSpaceID dGeomGetSpace (dGeomID)
    int dGeomGetClass (dGeomID)
    
    # Add by Yulong Zhang
    int dGeomGetDrawAxisFlag (dGeomID geom)
    void dGeomSetDrawAxisFlag (dGeomID geom, int x)

    void dGeomSetCategoryBits(dGeomID, unsigned long bits)
    void dGeomSetCollideBits(dGeomID, unsigned long bits)
    unsigned long dGeomGetCategoryBits(dGeomID)
    unsigned long dGeomGetCollideBits(dGeomID)

    void dGeomEnable (dGeomID)
    void dGeomDisable (dGeomID)
    int dGeomIsEnabled (dGeomID)

    void dGeomGroupAdd (dGeomID group, dGeomID x)
    void dGeomGroupRemove (dGeomID group, dGeomID x)
    int dGeomGroupGetNumGeoms (dGeomID group)
    dGeomID dGeomGroupGetGeom (dGeomID group, int i)

    dGeomID dCreateGeomTransform (dSpaceID space)
    void dGeomTransformSetGeom (dGeomID g, dGeomID obj)
    dGeomID dGeomTransformGetGeom (dGeomID g)
    void dGeomTransformSetCleanup (dGeomID g, int mode)
    int dGeomTransformGetCleanup (dGeomID g)
    void dGeomTransformSetInfo (dGeomID g, int mode)
    int dGeomTransformGetInfo (dGeomID g)

    int dCollide (dGeomID o1, dGeomID o2, int flags, dContactGeom *contact, int skip)

    # Trimesh
    dTriMeshDataID dGeomTriMeshDataCreate()
    void dGeomTriMeshDataDestroy(dTriMeshDataID g)
    void dGeomTriMeshDataBuildSingle1 (dTriMeshDataID g, void* Vertices,
                                int VertexStride, int VertexCount,
                                void* Indices, int IndexCount,
                                int TriStride, void* Normals)

    void dGeomTriMeshDataBuildSimple(dTriMeshDataID g, dReal* Vertices, int VertexCount, unsigned int* Indices, int IndexCount)

    dGeomID dCreateTriMesh (dSpaceID space, dTriMeshDataID Data,
                            void* Callback,
                            void* ArrayCallback,
                            void* RayCallback)

    void dGeomTriMeshSetData (dGeomID g, dTriMeshDataID Data)

    void dGeomTriMeshClearTCCache (dGeomID g)

    void dGeomTriMeshGetTriangle (dGeomID g, int Index, dVector3 *v0, dVector3 *v1, dVector3 *v2)

    int dGeomTriMeshGetTriangleCount (dGeomID g)

    void dGeomTriMeshGetPoint (dGeomID g, int Index, dReal u, dReal v, dVector3 Out)

    void dGeomTriMeshEnableTC(dGeomID g, int geomClass, int enable)
    int dGeomTriMeshIsTCEnabled(dGeomID g, int geomClass)

    # Heightfield
    dHeightfieldDataID dGeomHeightfieldDataCreate()
    void dGeomHeightfieldDataDestroy(dHeightfieldDataID g)
    void dGeomHeightfieldDataBuildCallback(dHeightfieldDataID d,
                                           void* pUserData,
                                           dHeightfieldGetHeight* pCallback,
                                           dReal width, dReal depth,
                                           int widthSamples, int depthSamples,
                                           dReal scale, dReal offset,
                                           dReal thickness, int bWrap)
    dGeomID dCreateHeightfield (dSpaceID space, dHeightfieldDataID data, int bPlaceable)

    # Add by Zhenhua Song
    void dNormalize3(dVector3)
    void dNormalize4(dVector4)
    dReal dCalcVectorLength3(const dReal *)
    dReal dCalcVectorLengthSquare3(const dReal *)

    # Add by Zhenhua Song
    void ODEMat3ToDenseMat3(const dMatrix3 odeMat3, dReal * denseMat3Out, int offset)

    # Add by Zhenhua Song
    void DenseMat3ToODEMat3(dMatrix3 odeMat3, const dReal * denseMat3In, int offset)

    # Add by Zhenhua Song
    void dSolveLCPWrapper (int n, dReal *A, dReal *x, dReal *b,
                           dReal *outer_w, int nub, dReal *lo, dReal *hi, int *findex)

    # Add by Zhenhua Song
    void dRandSetSeed(unsigned long s)

# end ode.h

# Add by Zhenhua Song

ctypedef struct dJointGroupWithdWorld:
    int use_max_force_contact
    int max_contact_num
    dReal soft_cfm
    dReal soft_erp
    int use_soft_contact
    int self_collision
    dJointGroupID group
    dWorldID world


# Add by Zhenhua Song
cdef extern from "joint_local_quat_batch.h":
    void ode_quat_to_scipy(dReal* q_ode)
    void scipy_quat_to_ode(dReal* q_scipy)
    void ode_quat_inv(dReal* ode_quat)
    void quat_arr_from_ode_to_scipy(dReal* qs, int count)
    void quat_arr_from_scipy_to_ode(dReal* qs, int count)
    dReal quat_dot(const dReal* q0, const dReal* q1)
    void minus_quat(dReal* q)
    void flip_ode_quat_by_w(dReal* ode_quat)
    void ode_quat_apply(const dReal* quat_ode, const dReal* vec_in, dReal* vec_out)

    void ode_quat_to_axis_angle(const dReal* ode_quat_input, dReal* result)

    void get_joint_local_quat_batch(
        dJointID* joints,
        int joint_count,
        dReal* parent_qs,
        dReal* child_qs,
        dReal* local_qs,
        dReal* parent_qs_inv,
        int convert_to_scipy)

    dReal compute_total_power_by_global(dJointID * joints, int joint_count, const dReal * global_joint_torques)
    dReal compute_total_power(dJointID * joints, int joint_count, const dReal * joint_torques)

    # As (stable) PD controller in python is too slow (by profile),
    # rewrite it in c++, then call it via cython
    void pd_control_batch(
        dJointID* joints,
        int joint_count,
        const dReal* input_target_local_qs,
        const dReal* kps,
        const dReal* kds,
        const dReal* torque_limits,
        dReal* local_res_joint_torques,
        dReal* global_res_joint_torques,
        int input_in_scipy
    )

cdef extern from "QuaternionWithGrad.h" nogil:
    void quat_multiply_single(
        const double * q1,
        const double * q2,
        double * q
    )

    void quat_inv_impl(
        const double * q,
        double * out_q,
        size_t num_quat
    )

    void quat_inv_single(
        const double * q,
        double * out_q
    )

    void quat_multiply_forward(
        const double * q1,
        const double * q2,
        double * q,
        size_t num_quat
    )

    void quat_multiply_backward_single(
        const double * q1,
        const double * q2,
        const double * grad_q, # \frac{\partial L}{\partial q_x, q_y, q_z, q_w}
        double * grad_q1,
        double * grad_q2
    )

    void quat_multiply_backward(
        const double * q1,
        const double * q2,
        const double * grad_q,
        double * grad_q1,
        double * grad_q2,
        size_t num_quat
    )

    void quat_apply_single(
        const double * q,
        const double * v,
        double * o
    )

    void quat_apply_forward(
        const double * q,
        const double * v,
        double * o,
        size_t num_quat
    )

    # Add by Yulong Zhang
    void quat_apply_forward_one2many(
        const double * q,
        const double * v,
        double * o,
        size_t num_quat
    )

    void quat_apply_backward_single(
        const double * q,
        const double * v,
        const double * o_grad,
        double * q_grad,
        double * v_grad
    )

    void quat_apply_backward(
        const double * q,
        const double * v,
        const double * o_grad,
        double * q_grad,
        double * v_grad,
        size_t num_quat
    )

    void flip_quat_by_w_forward_impl(
        const double * q,
        double * q_out,
        size_t num_quat
    )

    void flip_quat_by_w_backward_impl(
        const double * q,
        const double * grad_in,
        double * grad_out,
        size_t num_quat
    )

    void quat_to_vec6d_single(
        const double * q,
        double * vec6d
    )

    void quat_to_vec6d_impl(const double * q, double * vec6d, size_t num_quat)

    void quat_to_matrix_forward_single(
        const double * q,
        double * mat
    )

    void quat_to_matrix_impl(
        const double * q,
        double * mat,
        size_t num_quat
    )

    void quat_to_matrix_backward_single(
        const double * q,
        const double * grad_in,
        double * grad_out
    )

    void quat_to_matrix_backward(
        const double * q,
        const double * grad_in,
        double * grad_out,
        size_t num_quat
    )
    # Add by Yulong Zhang
    void six_dim_mat_to_quat_single(
        const double * mat,
        double * quat
    )
    void six_dim_mat_to_quat_impl(
        const double * mat,
        double * q,
        size_t num_quat
    )
    void vector_to_cross_matrix_single(
        const double * vec,
        double * mat
    )

    void vector_to_cross_matrix_impl(
        const double * vec,
        double * mat,
        size_t num_vec
    )

    void vector_to_cross_matrix_backward_single(
        const double * vec,
        const double * grad_in,
        double * grad_out
    )

    void vector_to_cross_matrix_backward(
        const double * vec,
        const double * grad_in,
        double * grad_out,
        size_t num_vec
    )

    void quat_to_rotvec_single(
        const double * q,
        double & angle,
        double * rotvec
    )

    void quat_to_rotvec_impl(
        const double * q,
        double * angle,
        double * rotvec,
        size_t num_quat
    )

    void quat_to_rotvec_backward_single(
        const double * q,
        double angle,
        const double * grad_in,
        double * grad_out
    )

    void quat_to_rotvec_backward(
        const double * q,
        const double * angle,
        const double * grad_in,
        double * grad_out,
        size_t num_quat
    )

    void quat_from_rotvec_single(
        const double * rotvec,
        double * q
    )

    void quat_from_rotvec_impl(
        const double * rotvec,
        double * q,
        size_t num_quat
    )

    void quat_from_rotvec_backward_single(
        const double * rotvec,
        const double * grad_in,
        double * grad_out
    )

    void quat_from_rotvec_backward_impl(
        const double * rotvec,
        const double * grad_in,
        double * grad_out,
        size_t num_quat
    )

    void quat_from_matrix_single(
        const double * mat,
        double * q
    )

    void quat_from_matrix_impl(
        const double * mat,
        double * q,
        size_t num_quat
    )

    void quat_from_matrix_backward_single(
        const double * mat,
        const double * grad_in,
        double * grad_out
    )

    void quat_from_matrix_backward_impl(
        const double * mat,
        const double * grad_in,
        double * grad_out,
        size_t num_quat
    )

    void quat_to_hinge_angle_single(
        const double * q,
        const double * axis,
        double & angle
    )

    void quat_to_hinge_angle_forward(
        const double * q,
        const double * axis,
        double * angle,
        size_t num_quat
    )

    void quat_to_hinge_angle_backward_single(
        const double * q,
        const double * axis,
        double grad_in,
        double * grad_out
    )

    void quat_to_hinge_angle_backward(
        const double * q,
        const double * axis,
        const double * grad_in,
        double * grad_out,
        size_t num_quat
    )

    void parent_child_quat_to_hinge_angle(
        const double * quat0,
        const double * quat1,
        const double * init_rel_quat_inv,
        const double * axis,
        double * angle,
        size_t num_quat
    )

    void parent_child_quat_to_hinge_angle_backward(
        const double * quat0,
        const double * quat1,
        const double * init_rel_quat_inv,
        const double * axis,
        const double * grad_in,
        double * quat0_grad,
        double * quat1_grad,
        size_t num_quat
    )

    void quat_integrate_impl(
        const double * q,
        const double * omega,
        double dt,
        double * result,
        size_t num_quat
    )

    void quat_integrate_backward(
        const double * q,
        const double * omega,
        double dt,
        const double * grad_in,
        double * q_grad,
        double * omega_grad,
        size_t num_quat
    )

    void vector_normalize_single(
        const double * x,
        size_t ndim,
        double * result
    )

    void vector_normalize_backward_single(
        const double * x,
        size_t ndim,
        const double * grad_in,
        double * grad_out
    )

    void quat_integrate_single(
        const double * q,
        const double * omega,
        double dt,
        double * result
    )

    void quat_integrate_impl(
        const double * q,
        const double * omega,
        double dt,
        double * result,
        size_t num_quat
    )

    void quat_integrate_backward_single(
        const double * q,
        const double * omega,
        double dt,
        const double * grad_in,
        double * q_grad,
        double * omega_grad
    )

    void quat_integrate_backward(
        const double * q,
        const double * omega,
        double dt,
        const double * grad_in,
        double * q_grad,
        double * omega_grad,
        size_t num_quat
    )

    # Add by Yulong Zhang
    void calc_surface_distance_to_capsule(
        const double * relative_pos,
        size_t ndim,
        double radius,
        double length,
        double * sd,
        double * normal
    )

    void clip_vec_by_norm_forward_single(
        const double * x,
        double min_val,
        double max_val,
        double * result,
        size_t ndim
    )

    void clip_vec_by_norm_backward_single(
        const double * x,
        double min_val,
        double max_val,
        const double * grad_in,
        double * grad_out,
        size_t ndim
    )

    void clip_vec_by_length_forward(
        const double * x,
        double max_len,
        double * result,
        size_t ndim
    )

    void clip_vec3_arr_by_length_forward(
        const double * x,
        const double * max_len,
        double * result,
        size_t num_vecs
    )

    void clip_vec_by_length_backward(
        const double * x,
        double max_len,
        const double * grad_in,
        double * grad_out,
        size_t ndim
    )

    void clip_vec3_arr_by_length_backward(
        const double * x,
        const double * max_len,
        const double * grad_in,
        double * grad_out,
        size_t num_vecs
    )

    void decompose_rotation_single(
        const double * q,
        const double * vb,
        double * result
    )

    void decompose_rotation(
        const double * q,
        const double * v,
        double * result,
        size_t num_quat
    )

    void decompose_rotation_pair_single(
        const double * q,
        const double * vb,
        double * q_a,
        double * q_b
    )

    void decompose_rotation_pair(
        const double * q,
        const double * vb,
        double * q_a,
        double * q_b,
        size_t num_quat
    )

    void decompose_rotation_pair_one2many(
        const double * q,
        const double * vb,
        double * q_a,
        double * q_b,
        size_t num_quat
    )
    void decompose_rotation_backward_single(
        const double * q,
        const double * v,
        const double * grad_in,
        double * grad_q,
        double * grad_v
    )

    void decompose_rotation_backward(
        const double * q,
        const double * v,
        const double * grad_in,
        double * grad_q,
        double * grad_v,
        size_t num_quat
    )