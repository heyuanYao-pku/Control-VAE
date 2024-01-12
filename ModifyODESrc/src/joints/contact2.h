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

//////////////////////
// created by Libin Liu

#ifndef _ODE_JOINT_CONTACT_2_H_
#define _ODE_JOINT_CONTACT_2_H_

#include "joint.h"

// contact

struct dxJointContact2 : public dxJoint
{
    int the_m;   // number of rows computed by getInfo1
    int the_nub;
    int donotPushOut;
    dContact contact;

    dxJointContact2(dxWorld* w);
    virtual void getSureMaxInfo(SureMaxInfo* info);
    virtual void getInfo1(Info1* info);
    virtual void getInfo2(Info2* info);
    virtual dJointType type() const;
    virtual size_t size() const;


    ///////////
    // added by Libin
    virtual void getInfo3(Info3* info);
};


#endif

