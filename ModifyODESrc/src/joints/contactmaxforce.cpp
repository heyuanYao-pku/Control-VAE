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

#include "contactmaxforce.h"
#include "config.h"
#include "joint_internal.h"
#include <iostream>


dxJointContactMaxForce::dxJointContactMaxForce(dxWorld* w) : dxJointContact(w)
{

}

//dReal dxJointContactMaxForce::static_max_friction_value = dInfinity; // static..

/// <summary>
/// All of findex is set to -1. The contact mu means the maximal contact force..
/// 
/// </summary>
/// <param name="info"></param>
void dxJointContactMaxForce::getInfo2(dxJoint::Info2* info)
{
    // set the cfm value here
    info->cfm[0] = info->cfm[1] = info->cfm[2] = this->cfm;

    int s = info->rowskip;
    int s2 = 2 * s;

    // get normal, with sign adjusted for body1/body2 polarity
    dVector3 normal;
    dCopyVector3(normal, contact.geom.normal);
    normal[3] = 0; // Actually, normal[3] is not used

    // c1,c2 = contact points with respect to body PORs
    dVector3 c1, c2 = { 0,0,0 };
    dSubtractVectors3(c1, contact.geom.pos, node[0].body->posr.pos);

    // set jacobian for normal. supporting force is along normal vector
    dCopyVector3(info->J1l, normal);
    dCalcVectorCross3(info->J1a, c1, normal);
    if (node[1].body)
    {
        dSubtractVectors3(c2, contact.geom.pos, node[1].body->posr.pos);
        dCopyNegatedVector3(info->J2l, normal);
        dCalcVectorCross3(info->J2a, c2, normal);
        dNegateVector3(info->J2a);
    }

    // set right hand side and cfm value for normal
    // dReal erp = info->erp; // Modify by Zhenhua Song
    dReal erp = this->erp;
    if (contact.surface.mode & dContactSoftERP)
        erp = contact.surface.soft_erp;

    // for debug
    // std::cout << "erp = " << erp << std::endl;

    dReal k = info->fps * erp;
    dReal depth = contact.geom.depth;
    if (depth < 0) depth = 0; // always depth >= 0.

    if (contact.surface.mode & dContactSoftCFM) // Zhenhua Song: this is not used in our program
        info->cfm[0] = contact.surface.soft_cfm;

    const dReal pushout = k * depth;
    info->c[0] = pushout;

    // set LCP limits for normal
    info->lo[0] = 0; // length of supporting force >= 0
    info->hi[0] = dInfinity;

    // now do jacobian for tangential forces
    dVector3 t1, t2; // two vectors tangential to normal

    // first friction direction
    if (the_m >= 2)
    {
        dPlaneSpace(normal, t1, t2);
        dCopyVector3(info->J1l + s, t1);
        dCalcVectorCross3(info->J1a + s, c1, t1);
        if (node[1].body)
        {
            dCopyNegatedVector3(info->J2l + s, t1);
            dCalcVectorCross3(info->J2a + s, c2, t1);
            dNegateVector3(info->J2a + s);
        }
        // set LCP bounds and friction index. this depends on the approximation mode
        info->lo[1] = -contact.surface.mu;
        info->hi[1] = contact.surface.mu;

        if (contact.surface.mode & dContactSlip1)
        {
            info->cfm[1] = contact.surface.slip1;
        }
    }

    // second friction direction
    if (the_m >= 3)
    {
        dCopyVector3(info->J1l + s2, t2);
        dCalcVectorCross3(info->J1a + s2, c1, t2);
        if (node[1].body)
        {
            dCopyNegatedVector3(info->J2l + s2, t2);
            dCalcVectorCross3(info->J2a + s2, c2, t2);
            dNegateVector3(info->J2a + s2);
        }
        // set LCP bounds and friction index. this depends on the approximation mode
        info->lo[2] = -contact.surface.mu;
        info->hi[2] = contact.surface.mu;

        if (contact.surface.mode & dContactSlip2)
        {
            info->cfm[1] = contact.surface.slip2;
        }
    }

    // for debug
    //std::cout << "joint cfm = " << this->cfm << std::endl;
    //for (int i = 0; i < 3; i++)
    //{
    //    std::cout << "cfm[" << i << "] = " << info->cfm[i] << "    ";
    //}
    //std::cout << std::endl;
}

dJointType dxJointContactMaxForce::type() const
{
    return dJointTypeContactMaxForce;
}

size_t dxJointContactMaxForce::size() const
{
    return sizeof(*this);
}
