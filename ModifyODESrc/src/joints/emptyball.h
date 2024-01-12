#pragma once
#include "ball.h"

// Add by Zhenhua Song
struct dxJointEmptyBall : public dxJointBall
{
	dxJointEmptyBall(dxWorld* w);
    virtual void getSureMaxInfo(SureMaxInfo* info);
    virtual void getInfo1(Info1* info);
    virtual void getInfo2(Info2* info);
    virtual dJointType type() const;
    virtual size_t size() const;

    virtual void setRelativeValues();
};
