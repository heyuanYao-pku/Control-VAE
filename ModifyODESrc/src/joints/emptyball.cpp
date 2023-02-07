#include "emptyball.h"
#include "config.h"
#include "joint_internal.h"

dxJointEmptyBall::dxJointEmptyBall(dxWorld* w) :
    dxJointBall(w)
{

}

void
dxJointEmptyBall::getSureMaxInfo(SureMaxInfo* info)
{
    info->max_m = 0;
}


void
dxJointEmptyBall::getInfo1(dxJoint::Info1* info)
{
    info->m = 0;
    info->nub = 0;
}


void
dxJointEmptyBall::getInfo2(dxJoint::Info2* info)
{
    dDebug(0, "dxJointEmptyBall::getInfo2 should never get called");
}

dJointType
dxJointEmptyBall::type() const
{
    return dJointTypeEmptyBall;
}

size_t
dxJointEmptyBall::size() const
{
    return sizeof(*this);
}

void dJointSetEmptyBallAnchor(dJointID j, dReal x, dReal y, dReal z)
{
    dxJointEmptyBall* joint = (dxJointEmptyBall *)j;
    dUASSERT(joint, "bad joint argument");
    checktype(joint, EmptyBall);
    setAnchors(joint, x, y, z, joint->anchor1, joint->anchor2);
}


void dJointSetEmptyBallAnchor2(dJointID j, dReal x, dReal y, dReal z)
{
    dxJointEmptyBall* joint = (dxJointEmptyBall*)j;
    dUASSERT(joint, "bad joint argument");
    checktype(joint, EmptyBall);
    joint->anchor2[0] = x;
    joint->anchor2[1] = y;
    joint->anchor2[2] = z;
    joint->anchor2[3] = 0;
}

void dJointGetEmptyBallAnchor(dJointID j, dVector3 result)
{
    dxJointEmptyBall* joint = (dxJointEmptyBall*)j;
    dUASSERT(joint, "bad joint argument");
    dUASSERT(result, "bad result argument");
    checktype(joint, EmptyBall);
    if (joint->flags & dJOINT_REVERSE)
        getAnchor2(joint, result, joint->anchor2);
    else
        getAnchor(joint, result, joint->anchor1);
}


void dJointGetEmptyBallAnchor2(dJointID j, dVector3 result)
{
    dxJointEmptyBall* joint = (dxJointEmptyBall*)j;
    dUASSERT(joint, "bad joint argument");
    dUASSERT(result, "bad result argument");
    checktype(joint, EmptyBall);
    if (joint->flags & dJOINT_REVERSE)
        getAnchor(joint, result, joint->anchor1);
    else
        getAnchor2(joint, result, joint->anchor2);
}

// Add by Zhenhua Song
const dReal* dJointGetEmptyBallAnchor1Raw(dJointID j)
{
    dxJointEmptyBall* joint = (dxJointEmptyBall*)j;
    dUASSERT(joint, "bad joint argument");
    checktype(joint, EmptyBall);
    if (joint->flags & dJOINT_REVERSE)
        return joint->anchor2;
    else
        return joint->anchor1;
}

// Add by Zhenhua Song
const dReal* dJointGetEmptyBallAnchor2Raw(dJointID j)
{
    dxJointEmptyBall* joint = (dxJointEmptyBall*)j;
    dUASSERT(joint, "bad joint argument");
    checktype(joint, EmptyBall);
    if (joint->flags & dJOINT_REVERSE)
        return joint->anchor1;
    else
        return joint->anchor2;
}

void dJointSetEmptyBallParam(dJointID j, int parameter, dReal value)
{
    dxJointEmptyBall* joint = (dxJointEmptyBall*)j;
    dUASSERT(joint, "bad joint argument");
    checktype(joint, EmptyBall);
    joint->set(parameter, value);
}


dReal dJointGetEmptyBallParam(dJointID j, int parameter)
{
    dxJointEmptyBall* joint = (dxJointEmptyBall*)j;
    dUASSERT(joint, "bad joint argument");
    checktype(joint, EmptyBall);
    return joint->get(parameter);
}

void
dxJointEmptyBall::setRelativeValues()
{
    dVector3 anchor;
    dJointGetEmptyBallAnchor(this, anchor);
    setAnchors(this, anchor[0], anchor[1], anchor[2], anchor1, anchor2);
}
