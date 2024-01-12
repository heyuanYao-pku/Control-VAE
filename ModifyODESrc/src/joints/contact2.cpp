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


#include "config.h"
#include "contact2.h"
#include "joint_internal.h"


#ifndef M_SQRT2
#define M_SQRT2    REAL(1.4142135623730950488016887242097)
#endif

#ifndef M_SQRT3
#define M_SQRT3    REAL(1.7320508075688772935274463415059)
#endif

#ifndef M_SQRT3_2
#define M_SQRT3_2    REAL(0.86602540378443864676372317075294)
#endif

dxJoint * dJointCreateContact2(dWorldID w, dJointGroupID group,
    const dContact *c, int donotPushOut)
{
    dAASSERT(w && c);
    dxJoint *j0;
    if (group) {
        j0 = (dxJointContact2*)group->stack.alloc(sizeof(dxJointContact2));
        group->num++;
    }
    else
        j0 = (dxJointContact2*)dAlloc(sizeof(dxJointContact2));

    dxJointContact2 *j = new(j0) dxJointContact2(w);
    if (group)
        j->flags |= dJOINT_INGROUP;

    j->donotPushOut = donotPushOut;
    j->contact = *c;
    return j;
}

//****************************************************************************
// contact

dxJointContact2::dxJointContact2(dxWorld *w)
    : dxJoint(w)
{
}


void
dxJointContact2::getSureMaxInfo(SureMaxInfo* info)
{
    info->max_m = 8; // 3 if mu == dInfinity, else 8
}


void
dxJointContact2::getInfo1(dxJoint::Info1 *info)
{
    // make sure mu's >= 0, then calculate number of constraint rows and number
    // of unbounded rows.
    int m = 1, nub = 0;
    if (contact.surface.mu < 0) 
        contact.surface.mu = 0;    
    if (contact.surface.mu == dInfinity)
    {
        m += 2;
        nub += 2;
    }
    else if (contact.surface.mu > 0)
    {
        m = 4;
        nub = 0;
    }
    
    the_m = m;
    the_nub = nub;
    info->m = m;
    info->nub = nub;
}


void
dxJointContact2::getInfo2(dxJoint::Info2 *info)
{
    int s = info->rowskip;
    dReal fps = info->fps;

    // get normal, with sign adjusted for body1/body2 polarity
    dVector3 normal;
    if (flags & dJOINT_REVERSE)
    {
        normal[0] = -contact.geom.normal[0];
        normal[1] = -contact.geom.normal[1];
        normal[2] = -contact.geom.normal[2];
    }
    else
    {
        normal[0] = contact.geom.normal[0];
        normal[1] = contact.geom.normal[1];
        normal[2] = contact.geom.normal[2];
    }
    normal[3] = 0; // @@@ hmmm

    // c1,c2 = contact points with respect to body PORs
    dVector3 c1, c2 = { 0, 0, 0 };
    c1[0] = contact.geom.pos[0] - node[0].body->posr.pos[0];
    c1[1] = contact.geom.pos[1] - node[0].body->posr.pos[1];
    c1[2] = contact.geom.pos[2] - node[0].body->posr.pos[2];

    // set jacobian for normal
    info->J1l[0] = normal[0];
    info->J1l[1] = normal[1];
    info->J1l[2] = normal[2];
    dCalcVectorCross3(info->J1a, c1, normal);
    if (node[1].body)
    {
        c2[0] = contact.geom.pos[0] - node[1].body->posr.pos[0];
        c2[1] = contact.geom.pos[1] - node[1].body->posr.pos[1];
        c2[2] = contact.geom.pos[2] - node[1].body->posr.pos[2];
        info->J2l[0] = -normal[0];
        info->J2l[1] = -normal[1];
        info->J2l[2] = -normal[2];
        dCalcVectorCross3(info->J2a, c2, normal);
        dNegateVector3(info->J2a);
    }

    // set right hand side and cfm value for normal
    dReal erp = info->erp;
    if (contact.surface.mode & dContactSoftERP)
        erp = contact.surface.soft_erp;
    dReal k = info->fps * erp;
    dReal depth = contact.geom.depth - world->contactp.min_depth;
    if (depth < 0) depth = 0;
    dReal motionN = 0;
    if (contact.surface.mode & dContactMotionN)
        motionN = contact.surface.motionN;

    const dReal pushout = k * depth + motionN;

    if (the_m == 1 || (the_m <= 3 && the_nub > 0))
    {
        if (contact.surface.mode & dContactSoftCFM)
            info->cfm[0] = contact.surface.soft_cfm;

        if (donotPushOut)
            info->c[0] = 0;
        else
        {
            info->c[0] = pushout;

            // note: this cap should not limit bounce velocity
            const dReal maxvel = world->contactp.max_vel;
            if (info->c[0] > maxvel)
                info->c[0] = maxvel;

            // deal with bounce
            if (contact.surface.mode & dContactBounce)
            {
                // calculate outgoing velocity (-ve for incoming contact)
                dReal outgoing = dCalcVectorDot3(info->J1l, node[0].body->lvel)
                    + dCalcVectorDot3(info->J1a, node[0].body->avel);
                if (node[1].body)
                {
                    outgoing += dCalcVectorDot3(info->J2l, node[1].body->lvel)
                        + dCalcVectorDot3(info->J2a, node[1].body->avel);
                }
                outgoing -= motionN;
                // only apply bounce if the outgoing velocity is greater than the
                // threshold, and if the resulting c[0] exceeds what we already have.
                if (contact.surface.bounce_vel >= 0 &&
                    (-outgoing) > contact.surface.bounce_vel)
                {
                    dReal newc = -contact.surface.bounce * outgoing + motionN;
                    if (newc > info->c[0]) info->c[0] = newc;
                }
            }
        }

        // set LCP limits for normal
        info->lo[0] = 0;
        info->hi[0] = dInfinity;

        if (!donotPushOut && info->addJdot)
        {
            // J vdot + Jdotv >= c
            // J (vnext - vn)/h + Jdot vn >= c
            // J vnext >= h*c + J*vn - h*Jdot*vn
            dVector3 tmp;
            dReal bias = 0;
            dCalcVectorCross3(tmp, node[0].body->avel, c1);
            dCalcVectorCross3(tmp, tmp, normal);
            bias += dCalcVectorDot3(tmp, node[0].body->avel) / fps;
            bias += dCalcVectorDot3(info->J1l, node[0].body->lvel);
            bias += dCalcVectorDot3(info->J1a, node[0].body->avel);
            if (node[1].body)
            {
                dCalcVectorCross3(tmp, node[1].body->avel, c2);
                dCalcVectorCross3(tmp, tmp, normal);
                bias -= dCalcVectorDot3(tmp, node[1].body->avel) / fps;
                bias += dCalcVectorDot3(info->J2l, node[1].body->lvel);
                bias += dCalcVectorDot3(info->J2a, node[1].body->avel);
            }
            info->c[0] += bias;
        }

        // now do jacobian for tangential forces
        dVector3 t1, t2; // two vectors tangential to normal

        // first friction direction
        if (the_nub > 0) // the_nub == 2
        {
            // by assumption, we have to make unbounded constriants in front of 
            // any bounded (lcp) constraints
            int s2 = s * 2;
            info->J1l[s2 + 0] = info->J1l[0];
            info->J1l[s2 + 1] = info->J1l[1];
            info->J1l[s2 + 2] = info->J1l[2];
            info->J1a[s2 + 0] = info->J1a[0];
            info->J1a[s2 + 1] = info->J1a[1];
            info->J1a[s2 + 2] = info->J1a[2];
            info->J2l[s2 + 0] = info->J2l[0];
            info->J2l[s2 + 1] = info->J2l[1];
            info->J2l[s2 + 2] = info->J2l[2];
            info->J2a[s2 + 0] = info->J2a[0];
            info->J2a[s2 + 1] = info->J2a[1];
            info->J2a[s2 + 2] = info->J2a[2];
            info->c[2] = info->c[0];
            info->cfm[2] = info->cfm[0];
            info->lo[2] = info->lo[0];
            info->hi[2] = info->hi[0];
            info->findex[2] = info->findex[0];

            // first direction
            {
                // identical to contact class
                if (contact.surface.mode & dContactFDir1)   // use fdir1 ?
                {
                    t1[0] = contact.fdir1[0];
                    t1[1] = contact.fdir1[1];
                    t1[2] = contact.fdir1[2];
                    dCalcVectorCross3(t2, normal, t1);
                }
                else
                {
                    dPlaneSpace(normal, t1, t2);
                }
                info->J1l[0] = t1[0];
                info->J1l[1] = t1[1];
                info->J1l[2] = t1[2];
                dCalcVectorCross3(info->J1a, c1, t1);
                if (node[1].body)
                {
                    info->J2l[0] = -t1[0];
                    info->J2l[1] = -t1[1];
                    info->J2l[2] = -t1[2];
                    dCalcVectorCross3(info->J2a, c2, t1);
                    dNegateVector3(info->J2a);
                }
                // set right hand side
                if (contact.surface.mode & dContactMotion1)
                {
                    info->c[0] = contact.surface.motion1;
                }
                info->c[0] = 0;
                if (contact.surface.mode & dContactSoftCFM)
                    info->cfm[0] = contact.surface.soft_cfm;
                else
                    info->cfm[0] = world->global_cfm;

                info->lo[0] = -dInfinity;
                info->hi[0] = dInfinity;
                info->findex[0] = -1;

                // set slip (constraint force mixing)
                if (contact.surface.mode & dContactSlip1)
                    info->cfm[0] = contact.surface.slip1;
            }

            // second direction
            {
                info->J1l[s + 0] = t2[0];
                info->J1l[s + 1] = t2[1];
                info->J1l[s + 2] = t2[2];
                dCalcVectorCross3(info->J1a + s, c1, t2);
                if (node[1].body)
                {
                    info->J2l[s + 0] = -t2[0];
                    info->J2l[s + 1] = -t2[1];
                    info->J2l[s + 2] = -t2[2];
                    dReal *J2a_plus_s = info->J2a + s;
                    dCalcVectorCross3(J2a_plus_s, c2, t2);
                    dNegateVector3(J2a_plus_s);
                }
                // set right hand side
                if (contact.surface.mode & dContactMotion2)
                {
                    info->c[1] = contact.surface.motion2;
                }
                // set LCP bounds and friction index. this depends on the approximation
                // mode
                info->c[1] = 0;
                if (contact.surface.mode & dContactSoftCFM)
                    info->cfm[1] = contact.surface.soft_cfm;
                else
                    info->cfm[1] = world->global_cfm;

                info->lo[1] = -dInfinity;
                info->hi[1] = dInfinity;
                info->findex[1] = -1;

                // set slip (constraint force mixing)
                if (contact.surface.mode & dContactSlip2)
                    info->cfm[1] = contact.surface.slip2;
            }
        }
    }
    else if (the_m == 3 || the_m == 6)
    {
        dVector3 t1, t2, t3, t4, t5, t6;
        // first direction
        if (contact.surface.mode & dContactFDir1)   // use fdir1 ?
        {
            t1[0] = contact.fdir1[0];
            t1[1] = contact.fdir1[1];
            t1[2] = contact.fdir1[2];
            dCalcVectorCross3(t2, normal, t1);
        }
        else
        {
            dPlaneSpace(normal, t1, t2);
        }
        {
            dScaleVector3(t2, M_SQRT3_2);
            dCopyNegatedVector3(t3, t2);
            double t10 = t1[0] * 0.5;
            double t11 = t1[1] * 0.5;
            double t12 = t1[2] * 0.5;
            t2[0] -= t10;
            t2[1] -= t11;
            t2[2] -= t12;
            t3[0] -= t10;
            t3[1] -= t11;
            t3[2] -= t12;
        }

        if (the_m == 6)
        {
            dCopyNegatedVector3(t4, t1);
            dCopyNegatedVector3(t5, t2);
            dCopyNegatedVector3(t6, t3);
        }

        dReal mu = contact.surface.mu;
        dReal scale_n = dRecipSqrt(dReal(1.0) + mu * mu);
        dReal scale_0 = mu * scale_n;
        dReal *ts[] = { t1, t2, t3, t4, t5, t6 };

        dScaleVector3(normal, scale_n);
        for (unsigned int i = 0; i < 3; ++i)
        {
            dScaleVector3(ts[i], scale_0);
            dAddVectors3(ts[i], ts[i], normal);
        }
        if (the_m == 8)
        {
            for (unsigned int i = 3; i < 6; ++i)
            {
                dScaleVector3(ts[i], scale_0);
                dAddVectors3(ts[i], ts[i], normal);
            }
        }
        int ss[] = { 0, s, s * 2, s * 3, s * 4, s * 5 };

        for (int i = 0; i < the_m; ++i)
        {
            int si = ss[i];
            dReal *ti = ts[i];
            info->J1l[si + 0] = ti[0];
            info->J1l[si + 1] = ti[1];
            info->J1l[si + 2] = ti[2];
            dCalcVectorCross3(info->J1a + si, c1, ti);
            if (node[1].body)
            {
                info->J2l[si + 0] = -ti[0];
                info->J2l[si + 1] = -ti[1];
                info->J2l[si + 2] = -ti[2];
                dCalcVectorCross3(info->J2a + si, c2, ti);
                dNegateVector3(info->J2a + si);
            }
            // set right hand side
            info->c[i] = 0;
            if (contact.surface.mode & dContactSoftCFM)
                info->cfm[i] = contact.surface.soft_cfm;
            else
                info->cfm[i] = world->global_cfm;
            info->lo[i] = 0;
            info->hi[i] = contact.surface.mu;
            info->findex[i] = -1;
        }
    }
    else if (the_m == 4 || the_m == 8)
    {
        dVector3 t1, t2, t3, t4, t5, t6, t7, t8;
        // first direction
        if (contact.surface.mode & dContactFDir1)   // use fdir1 ?
        {
            t1[0] = contact.fdir1[0];
            t1[1] = contact.fdir1[1];
            t1[2] = contact.fdir1[2];
            dCalcVectorCross3(t2, normal, t1);
        }
        else
        {
            dPlaneSpace(normal, t1, t2);
        }

        if (the_m == 8)
        {
            // t3 is the bisector of t1, t2
            dAddVectors3(t5, t1, t2);
            dScaleVector3(t5, M_SQRT1_2);
            // t4 is the bisector of t1, -t2
            dSubtractVectors3(t6, t1, t2);
            dScaleVector3(t6, M_SQRT1_2);
        }

        dReal mu = contact.surface.mu;
        dReal scale_n = dRecipSqrt(dReal(1.0) + mu * mu);
        dReal scale_0 = mu * scale_n;
        dReal *ts[] = { t1, t2, t3, t4, t5, t6, t7, t8 };

        dScaleVector3(normal, scale_n);
        for (unsigned int i = 0; i < 2; ++i)
        {
            dScaleVector3(ts[i], scale_0);
            dSubtractVectors3(ts[i + 2], normal, ts[i]);
            dAddVectors3(ts[i], ts[i], normal);
        }
        if (the_m == 8)
        {
            for (unsigned int i = 4; i < 6; ++i)
            {
                dScaleVector3(ts[i], scale_0);
                dSubtractVectors3(ts[i + 2], normal, ts[i]);
                dAddVectors3(ts[i], ts[i], normal);
            }
        }
        int ss[] = { 0, s, s * 2, s * 3, s * 4, s * 5, s * 6, s * 7 };

        for (int i = 0; i < the_m; ++i)
        {
            int si = ss[i];
            dReal *ti = ts[i];
            info->J1l[si + 0] = ti[0];
            info->J1l[si + 1] = ti[1];
            info->J1l[si + 2] = ti[2];
            dCalcVectorCross3(info->J1a + si, c1, ti);
            if (node[1].body)
            {
                info->J2l[si + 0] = -ti[0];
                info->J2l[si + 1] = -ti[1];
                info->J2l[si + 2] = -ti[2];
                dCalcVectorCross3(info->J2a + si, c2, ti);
                dNegateVector3(info->J2a + si);
            }
            // set right hand side
            info->c[i] = 0;
            if (contact.surface.mode & dContactSoftCFM)
                info->cfm[i] = contact.surface.soft_cfm;
            else
                info->cfm[i] = world->global_cfm;
            info->lo[i] = 0;
            info->hi[i] = contact.surface.mu;
            info->findex[i] = -1;
        }
    }
}

void 
dxJointContact2::getInfo3(Info3* info)
{
}


dJointType
dxJointContact2::type() const
{
    return dJointTypeContact2;
}


size_t
dxJointContact2::size() const
{
    return sizeof(*this);
}

