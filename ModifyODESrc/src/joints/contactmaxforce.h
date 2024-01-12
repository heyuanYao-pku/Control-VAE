#pragma once
#include "contact.h"
/// <summary>
/// Contact Joint should keep friction <= mu * (support force)
/// for simple, the formula is simplified as follow:
/// 0 <= support force <= +\infty
/// friction 0 <= contact mu (or max friction)
/// friction 1 <= contact mu (or max friction)
/// </summary>

struct dxJointContactMaxForce : public dxJointContact
{
	dxJointContactMaxForce(dxWorld* w);
	virtual void getInfo2(Info2* info);
	virtual dJointType type() const;
	virtual size_t size() const;
};
