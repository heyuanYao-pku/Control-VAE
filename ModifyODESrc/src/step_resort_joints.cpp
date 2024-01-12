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

#include <ode/odeconfig.h>
#include <ode/odemath.h>
#include <ode/rotation.h>
#include <ode/timer.h>
#include <ode/error.h>
#include <ode/matrix.h>
#include "config.h"
#include "objects.h"
#include "joints/joint.h"
#include "lcp.h"
#include "util.h"

#include <ode/extutils.h> // Add by Zhenhua Song

void dInternalStepIslandResortJoint_x2(dxWorldProcessMemArena* memarena,
    dxWorld* world, dxBody* const* body, unsigned int nb,
    dxJoint* const* _joint, unsigned int _nj, dReal stepsize, WorldStepFeatureInfoPtr feature_info)
{
	for (unsigned int i = 0; i < nb; ++i) body[i]->tag = i;

    const size_t ji_reserve_count = 2 * (size_t)_nj;
    dJointWithInfo1* jointiinfos = memarena->AllocateArray<dJointWithInfo1>(ji_reserve_count);
    unsigned int nub;
    size_t ji_start, ji_end;

    {
        size_t unb_start, mix_start, mix_end, lcp_end;
        unb_start = mix_start = mix_end = lcp_end = _nj;

        dJointWithInfo1* jicurr = jointiinfos + lcp_end;
        dxJoint* const* const _jend = _joint + _nj;
        dxJoint* const* _jcurr = _joint;
        while (true) {
            // -------------------------------------------------------------------------
            // Switch to growing array forward
            {
                bool fwd_end_reached = false;
                dJointWithInfo1* jimixend = jointiinfos + mix_end;
                while (true) {	// jicurr=dest, _jcurr=src
                    if (_jcurr == _jend) {
                        lcp_end = jicurr - jointiinfos;
                        fwd_end_reached = true;
                        break;
                    }
                    dxJoint* j = *_jcurr++;
                    j->getInfo1(&jicurr->info);
                    dIASSERT(jicurr->info.m >= 0 && jicurr->info.m <= 6 && jicurr->info.nub >= 0 && jicurr->info.nub <= jicurr->info.m);
                    if (jicurr->info.m > 0) {
                        if (jicurr->info.nub == 0) { // A lcp info - a correct guess!!!
                            jicurr->joint = j;
                            ++jicurr;
                        }
                        else if (jicurr->info.nub < jicurr->info.m) { // A mixed case
                            if (unb_start == mix_start) { // no unbounded infos yet - just move to opposite side of mixed-s
                                unb_start = mix_start = mix_start - 1;
                                dJointWithInfo1* jimixstart = jointiinfos + mix_start;
                                jimixstart->info = jicurr->info;
                                jimixstart->joint = j;
                            }
                            else if (jimixend != jicurr) { // have to swap to the tail of mixed-s
                                dxJoint::Info1 tmp_info = jicurr->info;
                                *jicurr = *jimixend;
                                jimixend->info = tmp_info;
                                jimixend->joint = j;
                                ++jimixend; ++jicurr;
                            }
                            else { // no need to swap as there are no LCP info-s yet
                                jicurr->joint = j;
                                jimixend = jicurr = jicurr + 1;
                            }
                        }
                        else { // A purely unbounded case -- break out and proceed growing in opposite direction
                            unb_start = unb_start - 1;
                            dJointWithInfo1* jiunbstart = jointiinfos + unb_start;
                            jiunbstart->info = jicurr->info;
                            jiunbstart->joint = j;
                            lcp_end = jicurr - jointiinfos;
                            mix_end = jimixend - jointiinfos;
                            jicurr = jiunbstart - 1;
                            break;
                        }
                    }
                    else {
                        j->tag = -1;
                    }
                }
                if (fwd_end_reached) {
                    break;
                }
            }
            // -------------------------------------------------------------------------
            // Switch to growing array backward
            {
                bool bkw_end_reached = false;
                dJointWithInfo1* jimixstart = jointiinfos + mix_start - 1;
                while (true) {	// jicurr=dest, _jcurr=src
                    if (_jcurr == _jend) {
                        unb_start = (jicurr + 1) - jointiinfos;
                        mix_start = (jimixstart + 1) - jointiinfos;
                        bkw_end_reached = true;
                        break;
                    }
                    dxJoint* j = *_jcurr++;
                    j->getInfo1(&jicurr->info);
                    dIASSERT(jicurr->info.m >= 0 && jicurr->info.m <= 6 && jicurr->info.nub >= 0 && jicurr->info.nub <= jicurr->info.m);
                    if (jicurr->info.m > 0) {
                        if (jicurr->info.nub == jicurr->info.m) { // An unbounded info - a correct guess!!!
                            jicurr->joint = j;
                            --jicurr;
                        }
                        else if (jicurr->info.nub > 0) { // A mixed case
                            if (mix_end == lcp_end) { // no lcp infos yet - just move to opposite side of mixed-s
                                dJointWithInfo1* jimixend = jointiinfos + mix_end;
                                lcp_end = mix_end = mix_end + 1;
                                jimixend->info = jicurr->info;
                                jimixend->joint = j;
                            }
                            else if (jimixstart != jicurr) { // have to swap to the head of mixed-s
                                dxJoint::Info1 tmp_info = jicurr->info;
                                *jicurr = *jimixstart;
                                jimixstart->info = tmp_info;
                                jimixstart->joint = j;
                                --jimixstart; --jicurr;
                            }
                            else { // no need to swap as there are no unbounded info-s yet
                                jicurr->joint = j;
                                jimixstart = jicurr = jicurr - 1;
                            }
                        }
                        else { // A purely lcp case -- break out and proceed growing in opposite direction
                            dJointWithInfo1* jilcpend = jointiinfos + lcp_end;
                            lcp_end = lcp_end + 1;
                            jilcpend->info = jicurr->info;
                            jilcpend->joint = j;
                            unb_start = (jicurr + 1) - jointiinfos;
                            mix_start = (jimixstart + 1) - jointiinfos;
                            jicurr = jilcpend + 1;
                            break;
                        }
                    }
                    else {
                        j->tag = -1;
                    }
                }
                if (bkw_end_reached) {
                    break;
                }
            }
        }

        nub = (unsigned)(mix_start - unb_start);
        dIASSERT((size_t)(mix_start - unb_start) <= (size_t)UINT_MAX);
        ji_start = unb_start;
        ji_end = lcp_end;
    }

    memarena->ShrinkArray<dJointWithInfo1>(jointiinfos, ji_reserve_count, ji_end);
    jointiinfos += ji_start;
    unsigned int nj = (unsigned int)(ji_end - ji_start);
    dIASSERT((size_t)(ji_end - ji_start) <= (size_t)UINT_MAX);

    unsigned int m = 0;

    {
        unsigned int mcurr = 0;
        const dJointWithInfo1* jicurr = jointiinfos;
        const dJointWithInfo1* const jiend = jicurr + nj;
        for (unsigned int i = 0; jicurr != jiend; i++, ++jicurr) {
            jicurr->joint->tag = i;
            unsigned int jm = jicurr->info.m;
            mcurr += jm;
        }

        m = mcurr;
    }

    // Add by Zhenhua Song: copy joint tag(or copy resorted joint pointer..)
    {
        feature_info->joints.data = (dJointID*)malloc(sizeof(dJointID) * nj); // TODO: memory release
        feature_info->joints.m = (unsigned int*)malloc(sizeof(unsigned int) * nj);
        feature_info->joints.nub = (unsigned int*)malloc(sizeof(unsigned int) * nj);
        feature_info->joints.cnt = nj;

        for (unsigned int i = 0; i < nj; i++)
        {
            feature_info->joints.data[i] = jointiinfos[i].joint;
            feature_info->joints.m[i] = jointiinfos[i].info.m;
            feature_info->joints.nub[i] = jointiinfos[i].info.nub;
        }

        // We don't need to save body resorted result. because resort M means M' = P * M, where P is resort matrix.
        // if we resort J by P, that is J' = J P^T.
        // then J' Minv' J'T = J P^T P Minv P^T P J^T. Result doesn't contains P.

        // if we want to resort Jacobian matrix by joint, we can write as J' = Q * J.
    }
}

void dInternalStepIslandResortJoint(dxWorldProcessMemArena* memarena,
    dxWorld* world, dxBody* const* body, unsigned int nb,
    dxJoint* const* joint, unsigned int nj, dReal stepsize, WorldStepFeatureInfoPtr feature_info)
{
    dInternalStepIslandResortJoint_x2(memarena, world, body, nb, joint, nj, stepsize, feature_info);
}
