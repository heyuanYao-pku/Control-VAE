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
//****************************************************************************
// misc defines

//#define TIMING
#if _DEBUG
#define DebugPrint 1
#else
#define DebugPrint 0
#endif


#ifdef TIMING
#define IFTIMING(x) x
#else
#define IFTIMING(x) ((void)0)
#endif



static void dInternalStepIslandWithInfo_x2(dxWorldProcessMemArena* memarena,
    dxWorld* world, dxBody* const* body, unsigned int nb,
    dxJoint* const* _joint, unsigned int _nj, dReal stepsize, WorldStepFeatureInfoPtr feature_info)
{
    IFTIMING(dTimerStart("preprocessing"));

    const dReal stepsizeRecip = dRecip(stepsize);

    {
        // number all bodies in the body list - set their tag values
        for (unsigned int i = 0; i < nb; ++i) body[i]->tag = i;
    }

    // for all bodies, compute the inertia tensor and its inverse in the global
    // frame, and compute the rotational force and add it to the torque
    // accumulator. invI are vertically stacked 3x4 matrices, one per body.
    // @@@ check computation of rotational force.

    dReal* invI = memarena->AllocateArray<dReal>(3 * 4 * (size_t)nb);

    { // Identical to QuickStep
        dReal* invIrow = invI;
        dxBody* const* const bodyend = body + nb;
        for (dxBody* const* bodycurr = body; bodycurr != bodyend; invIrow += 12, ++bodycurr) {
            dMatrix3 tmp;
            dxBody* b = *bodycurr;

            // compute inverse inertia tensor in global frame
            dMultiply2_333(tmp, b->invI, b->posr.R);
            dMultiply0_333(invIrow, b->posr.R, tmp); // R * invI * R^T

            if (b->flags & dxBodyGyroscopic) {
                dMatrix3 I;
                // compute inertia tensor in global frame
                dMultiply2_333(tmp, b->mass.I, b->posr.R);
                dMultiply0_333(I, b->posr.R, tmp); // I = R * I * R^T
                // compute rotational force
                dMultiply0_331(tmp, I, b->avel); // tmp = I * \omega
                dSubtractVectorCross3(b->tacc, b->avel, tmp); // tau = I \dot{\omega} + \omega \times I_c \omega. get I \dot{\omega}
            }
        }
    }

    { // Identical to QuickStep
      // add the gravity force to all bodies
      // since gravity does normally have only one component it's more efficient
      // to run three loops for each individual component
        dxBody* const* const bodyend = body + nb;
        dReal gravity_x = world->gravity[0];
        if (gravity_x) {
            for (dxBody* const* bodycurr = body; bodycurr != bodyend; ++bodycurr) {
                dxBody* b = *bodycurr;
                if ((b->flags & dxBodyNoGravity) == 0) {
                    b->facc[0] += b->mass.mass * gravity_x;
                }
            }
        }
        dReal gravity_y = world->gravity[1];
        if (gravity_y) {
            for (dxBody* const* bodycurr = body; bodycurr != bodyend; ++bodycurr) {
                dxBody* b = *bodycurr;
                if ((b->flags & dxBodyNoGravity) == 0) {
                    b->facc[1] += b->mass.mass * gravity_y;
                }
            }
        }
        dReal gravity_z = world->gravity[2];
        if (gravity_z) {
            for (dxBody* const* bodycurr = body; bodycurr != bodyend; ++bodycurr) {
                dxBody* b = *bodycurr;
                if ((b->flags & dxBodyNoGravity) == 0) {
                    b->facc[2] += b->mass.mass * gravity_z;
                }
            }
        }
    }

    // get m = total constraint dimension, nub = number of unbounded variables.
    // create constraint offset array and number-of-rows array for all joints.
    // the constraints are re-ordered as follows: the purely unbounded
    // constraints, the mixed unbounded + LCP constraints, and last the purely
    // LCP constraints. this assists the LCP solver to put all unbounded
    // variables at the start for a quick factorization.
    //
    // joints with m=0 are inactive and are removed from the joints array
    // entirely, so that the code that follows does not consider them.
    // also number all active joints in the joint list (set their tag values).
    // inactive joints receive a tag value of -1.

    // Reserve twice as much memory and start from the middle so that regardless of 
    // what direction the array grows to there would be sufficient room available.
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
        feature_info->joints.data = (dJointID*) malloc(sizeof(dJointID) * nj); // TODO: memory release
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

    // this will be set to the force due to the constraints
    dReal* cforce = memarena->AllocateArray<dReal>((size_t)nb * 8);
    dSetZero(cforce, (size_t)nb * 8);

    // if there are constraints, compute cforce
    if (m > 0) {
        // create a constraint equation right hand side vector `c', a constraint
        // force mixing vector `cfm', and LCP low and high bound vectors, and an
        // 'findex' vector.
        dReal* lo, * hi, * J, * A, * rhs;
        int* findex;

        {
            unsigned int mlocal = m;

            lo = memarena->AllocateArray<dReal>(mlocal);
            dSetValue(lo, mlocal, -dInfinity);

            hi = memarena->AllocateArray<dReal>(mlocal);
            dSetValue(hi, mlocal, dInfinity);

            J = memarena->AllocateArray<dReal>(2 * 8 * (size_t)mlocal);
            dSetZero(J, 2 * 8 * (size_t)mlocal);

            findex = memarena->AllocateArray<int>(mlocal);
            for (unsigned int i = 0; i < mlocal; ++i) findex[i] = -1;

            unsigned int mskip = dPAD(mlocal);
            A = memarena->AllocateArray<dReal>(mlocal * (size_t)mskip);
            dSetZero(A, mlocal * (size_t)mskip);

            rhs = memarena->AllocateArray<dReal>(mlocal);
            dSetZero(rhs, mlocal);
        }

        // Put 'c' in the same memory as 'rhs' as they transit into each other
        dReal* c = rhs; rhs = NULL; // erase rhs pointer for now as it is not to be used yet

        BEGIN_STATE_SAVE(memarena, cfmstate) {
            dReal* cfm = memarena->AllocateArray<dReal>(m);
            dSetValue(cfm, m, world->global_cfm);

            dReal* JinvM = memarena->AllocateArray<dReal>(2 * 8 * (size_t)m); // shape of JinvM is same as J((2*m)x8)
            dSetZero(JinvM, 2 * 8 * (size_t)m);

            {
                IFTIMING(dTimerNow("create J")); // Actually, shape of J is m x (6*numBody). compress it to (2*m) x 8
                // get jacobian data from constraints. a (2*m)x8 matrix will be created
                // to store the two jacobian blocks from each constraint. it has this
                // format:
                //
                //   l l l 0 a a a 0  \    .
                //   l l l 0 a a a 0   }-- jacobian body 1 block for joint 0 (3 rows)
                //   l l l 0 a a a 0  /
                //   l l l 0 a a a 0  \    .
                //   l l l 0 a a a 0   }-- jacobian body 2 block for joint 0 (3 rows)
                //   l l l 0 a a a 0  /
                //   l l l 0 a a a 0  }--- jacobian body 1 block for joint 1 (1 row)
                //   l l l 0 a a a 0  }--- jacobian body 2 block for joint 1 (1 row)
                //   etc...
                //
                //   (lll) = linear jacobian data
                //   (aaa) = angular jacobian data
                //

                dxJoint::Info2 Jinfo;
                Jinfo.rowskip = 8;
                Jinfo.fps = stepsizeRecip;
                Jinfo.erp = world->global_erp;
                Jinfo.addJdot = 0;

                unsigned ofsi = 0;
                const dJointWithInfo1* jicurr = jointiinfos;
                const dJointWithInfo1* const jiend = jicurr + nj;
                for (; jicurr != jiend; ++jicurr) {
                    const unsigned int infom = jicurr->info.m;
                    dReal* const J1row = J + 2 * 8 * (size_t)ofsi;
                    Jinfo.J1l = J1row;
                    Jinfo.J1a = J1row + 4;
                    dReal* const J2row = J1row + 8 * (size_t)infom;
                    Jinfo.J2l = J2row;
                    Jinfo.J2a = J2row + 4;
                    Jinfo.c = c + ofsi;
                    Jinfo.cfm = cfm + ofsi;
                    Jinfo.lo = lo + ofsi;
                    Jinfo.hi = hi + ofsi;
                    Jinfo.findex = findex + ofsi;

                    dxJoint* joint = jicurr->joint;
                    joint->getInfo2(&Jinfo); 
                    // comment by Zhenhua Song: usually, amotor limit will not be violated
                    // so, amotor joint will not appear here usually.

                    // adjust returned findex values for global index numbering
                    int* findex_ofsi = findex + ofsi;
                    for (unsigned int j = 0; j < infom; ++j) {
                        int fival = findex_ofsi[j];
                        if (fival != -1)
                            findex_ofsi[j] = fival + ofsi;
                    }

                    ofsi += infom;
                }
            }

            {
                // Add by Zhenhua Song
                feature_info->jacobian.data = (dReal*)malloc(sizeof(dReal) * 2 * m * 6); // TODO: change 8->6
                for (unsigned int row = 0; row < 2 * m; row++)
                {
                    for (unsigned int col = 0; col < 3; col++)
                    {
                        feature_info->jacobian.data[row * 6 + col] = J[row * 8 + col];
                    }
                    for (unsigned int col = 0; col < 3; col++)
                    {
                        feature_info->jacobian.data[row * 6 + 3 + col] = J[row * 8 + 4 + col];
                    }
                }
                feature_info->jacobian.row = 2 * m;
                feature_info->jacobian.column = 8;

                feature_info->joint_c.data = (dReal*)malloc(sizeof(dReal) * m);
                memcpy(feature_info->joint_c.data, c, sizeof(dReal) * m);
                feature_info->joint_c.row = m;
                feature_info->joint_c.column = 1;
            }

            {
                IFTIMING(dTimerNow("compute A"));
                {
                    // when not compressed, shape of J is m x (6*numBody), shape of invM is (6*numBody) x (6*numBody), so shape of A = J*invM*J^T is mxm.
                    // for memory align, shape of A is m x dPAD(m)
                    // 
                    // compute A = J*invM*J'. first compute JinvM = J*invM. this has the same
                    // format as J so we just go through the constraints in J multiplying by
                    // the appropriate scalars and matrices.
                    unsigned ofsi = 0;
                    const dJointWithInfo1* jicurr = jointiinfos;
                    const dJointWithInfo1* const jiend = jicurr + nj;
                    for (; jicurr != jiend; ++jicurr) {
                        const unsigned int infom = jicurr->info.m;
                        dxJoint* joint = jicurr->joint;
                        unsigned int b0 = joint->node[0].body->tag;
                        dReal body_invMass0 = body[b0]->invMass;
                        dReal* body_invI0 = invI + (size_t)b0 * 12;
                        dReal* Jsrc = J + 2 * 8 * (size_t)ofsi;
                        dReal* Jdst = JinvM + 2 * 8 * (size_t)ofsi;
                        for (unsigned int j = infom; j > 0;) {
                            j -= 1;
                            for (unsigned int k = 0; k < 3; ++k) Jdst[k] = Jsrc[k] * body_invMass0;
                            dMultiply0_133(Jdst + 4, Jsrc + 4, body_invI0); // divide J^a to (1 \times 3) row vectors, then multiply invI individually
                            Jsrc += 8;
                            Jdst += 8;
                        }

                        if (joint->node[1].body) {
                            unsigned int b1 = joint->node[1].body->tag;
                            dReal body_invMass1 = body[b1]->invMass;
                            dReal* body_invI1 = invI + (size_t)b1 * 12;
                            for (unsigned int j = infom; j > 0; ) {
                                j -= 1;
                                for (unsigned int k = 0; k < 3; ++k) Jdst[k] = Jsrc[k] * body_invMass1;
                                dMultiply0_133(Jdst + 4, Jsrc + 4, body_invI1);
                                Jsrc += 8;
                                Jdst += 8;
                            }
                        }

                        ofsi += infom;
                    }
                }

                {
                    // now compute A = JinvM * J'. A's rows and columns (m x dPad(m)) are grouped by joint,
                    // i.e. in the same way as the rows of J. block (i,j) of A is only nonzero
                    // if joints i and j have at least one body in common. 

                    BEGIN_STATE_SAVE(memarena, ofsstate) {
                        unsigned int* ofs = memarena->AllocateArray<unsigned int>(m);
                        const unsigned int mskip = dPAD(m);

                        unsigned ofsi = 0;
                        const dJointWithInfo1* jicurr = jointiinfos;
                        const dJointWithInfo1* const jiend = jicurr + nj;
                        for (unsigned int i = 0; jicurr != jiend; i++, ++jicurr) {
                            const unsigned int infom = jicurr->info.m;
                            dxJoint* joint = jicurr->joint;

                            dReal* Arow = A + mskip * (size_t)ofsi;
                            dReal* JinvMrow = JinvM + 2 * 8 * (size_t)ofsi;

                            dxBody* jb0 = joint->node[0].body;
                            for (dxJointNode* n0 = jb0->firstjoint; n0; n0 = n0->next) {
                                // if joint was tagged as -1 then it is an inactive (m=0 or disabled)
                                // joint that should not be considered
                                int j0 = n0->joint->tag;
                                if (j0 != -1 && (unsigned)j0 < i) {
                                    const dJointWithInfo1* jiother = jointiinfos + j0;
                                    size_t ofsother = (jiother->joint->node[1].body == jb0) ? 8 * (size_t)jiother->info.m : 0;
                                    // set block of A
                                    MultiplyAdd2_p8r(Arow + ofs[j0], JinvMrow,
                                        J + 2 * 8 * (size_t)ofs[j0] + ofsother, infom, jiother->info.m, mskip);
                                }
                            }

                            dxBody* jb1 = joint->node[1].body;
                            dIASSERT(jb1 != jb0);
                            if (jb1)
                            {
                                for (dxJointNode* n1 = jb1->firstjoint; n1; n1 = n1->next) {
                                    // if joint was tagged as -1 then it is an inactive (m=0 or disabled)
                                    // joint that should not be considered
                                    int j1 = n1->joint->tag;
                                    if (j1 != -1 && (unsigned)j1 < i) {
                                        const dJointWithInfo1* jiother = jointiinfos + j1;
                                        size_t ofsother = (jiother->joint->node[1].body == jb1) ? 8 * (size_t)jiother->info.m : 0;
                                        // set block of A
                                        MultiplyAdd2_p8r(Arow + ofs[j1], JinvMrow + 8 * (size_t)infom,
                                            J + 2 * 8 * (size_t)ofs[j1] + ofsother, infom, jiother->info.m, mskip);
                                    }
                                }
                            }

                            ofs[i] = ofsi;
                            ofsi += infom;
                        }

                    } END_STATE_SAVE(memarena, ofsstate);
                }

                {
                    // compute diagonal blocks of A
                    const unsigned int mskip = dPAD(m);

                    unsigned ofsi = 0;
                    const dJointWithInfo1* jicurr = jointiinfos;
                    const dJointWithInfo1* const jiend = jicurr + nj;
                    for (; jicurr != jiend; ++jicurr) {
                        const unsigned int infom = jicurr->info.m;
                        dReal* Arow = A + (mskip + 1) * (size_t)ofsi;
                        dReal* JinvMrow = JinvM + 2 * 8 * (size_t)ofsi;
                        dReal* Jrow = J + 2 * 8 * (size_t)ofsi;
                        Multiply2_p8r(Arow, JinvMrow, Jrow, infom, infom, mskip);
                        if (jicurr->joint->node[1].body) {
                            MultiplyAdd2_p8r(Arow, JinvMrow + 8 * (size_t)infom, Jrow + 8 * (size_t)infom, infom, infom, mskip);
                        }

                        ofsi += infom;
                    }
                }

                {
                    feature_info->j_minv_j.data = (dReal*)malloc(sizeof(dReal) * m * m);
                    int mskip = dPAD(m);
                    for (unsigned int i = 0; i < m; i++)
                    {
                        memcpy(feature_info->j_minv_j.data + i * m, A + i * mskip, sizeof(dReal)* m);
                    }
                }

                {
                    // add cfm to the diagonal of A, A = J^T M^{-1} J + 1/h CFM
                    const unsigned int mskip = dPAD(m);

                    dReal* Arow = A;
                    for (unsigned int i = 0; i < m; Arow += mskip, ++i) {
                        Arow[i] += cfm[i] * stepsizeRecip;
                    }

                    {
                        feature_info->cfm.data = (dReal*)malloc(sizeof(dReal) * m);
                        memcpy(feature_info->cfm.data, cfm, sizeof(dReal)* m);
                    }
#if DebugPrint
                    PrintMat("A", A, m, m, false, mskip);
#endif
                }
            }

        } END_STATE_SAVE(memarena, cfmstate);

        BEGIN_STATE_SAVE(memarena, tmp1state) {
            // compute the right hand side `rhs'
            IFTIMING(dTimerNow("compute rhs"));

            dReal* tmp1 = memarena->AllocateArray<dReal>((size_t)nb * 8);
            //dSetZero (tmp1,nb*8);

            {
                // put v/h + invM*fe into tmp1
                dReal* tmp1curr = tmp1;
                const dReal* invIrow = invI;
                dxBody* const* const bodyend = body + nb;
                for (dxBody* const* bodycurr = body; bodycurr != bodyend; tmp1curr += 8, invIrow += 12, ++bodycurr) {
                    dxBody* b = *bodycurr;
                    for (unsigned int j = 0; j < 3; ++j) tmp1curr[j] = b->facc[j] * b->invMass + b->lvel[j] * stepsizeRecip;
                    dMultiply0_331(tmp1curr + 4, invIrow, b->tacc);
                    for (unsigned int k = 0; k < 3; ++k) tmp1curr[4 + k] += b->avel[k] * stepsizeRecip;
                }
            }

            {
                // init rhs -- this erases 'c' as they reside in the same memory!!!
                rhs = c;
                for (unsigned int i = 0; i < m; ++i) rhs[i] = c[i] * stepsizeRecip;
                c = NULL; // set 'c' to NULL to prevent unexpected access
            }

            {
                // put J*tmp1 into rhs, rhs = c/h - J * (v/h + invM * fe)
                unsigned ofsi = 0;
                const dJointWithInfo1* jicurr = jointiinfos;
                const dJointWithInfo1* const jiend = jicurr + nj;
                for (; jicurr != jiend; ++jicurr) {
                    const unsigned int infom = jicurr->info.m;
                    dxJoint* joint = jicurr->joint;

                    dReal* rhscurr = rhs + ofsi;
                    const dReal* Jrow = J + 2 * 8 * (size_t)ofsi;
                    MultiplySub0_p81(rhscurr, Jrow, tmp1 + 8 * (size_t)(unsigned)joint->node[0].body->tag, infom);
                    if (joint->node[1].body) {
                        MultiplySub0_p81(rhscurr, Jrow + 8 * (size_t)infom, tmp1 + 8 * (size_t)(unsigned)joint->node[1].body->tag, infom);
                    }

                    ofsi += infom;
                }
            }
        } END_STATE_SAVE(memarena, tmp1state);

        dReal* lambda = memarena->AllocateArray<dReal>(m);

        BEGIN_STATE_SAVE(memarena, lcpstate) {
            IFTIMING(dTimerNow("solving LCP problem"));

            {
                feature_info->lcp_w.data = (dReal*)malloc(sizeof(dReal) * m); // DO NOT forget to memory release..

                feature_info->lcp_lo.data = (dReal*)malloc(sizeof(dReal) * m);
                memcpy(feature_info->lcp_lo.data, lo, sizeof(dReal) * m);
                feature_info->lcp_lo.row = m;
                feature_info->lcp_lo.column = 1;
                feature_info->lcp_lo.row_skip = 0;
                feature_info->lcp_lo.skip4 = 0;

                feature_info->lcp_hi.data = (dReal*)malloc(sizeof(dReal) * m);
                memcpy(feature_info->lcp_hi.data, hi, sizeof(dReal) * m);
                feature_info->lcp_hi.row = m;
                feature_info->lcp_hi.column = 1;
                feature_info->lcp_hi.row_skip = 0;
                feature_info->lcp_hi.skip4 = 0;

                feature_info->lcp_a.data = (dReal*)malloc(sizeof(dReal) * m * m);
                unsigned int mskip = dPAD(m);
                for (unsigned int row = 0; row < m; row++)
                {
                    memcpy((feature_info->lcp_a.data) + row * m, A + row * mskip, sizeof(dReal) * m);
                }
                feature_info->lcp_a.row = m;
                feature_info->lcp_a.column = m;
                feature_info->lcp_a.row_skip = m;
                feature_info->lcp_a.skip4 = 0;

                feature_info->lcp_rhs.data = (dReal*)malloc(sizeof(dReal) * m);
                memcpy(feature_info->lcp_rhs.data, rhs, sizeof(dReal) * m);
                feature_info->lcp_rhs.row = m;
                feature_info->lcp_rhs.column = 1;
                feature_info->lcp_rhs.row_skip = 0;
                feature_info->lcp_rhs.skip4 = 0;

                feature_info->findex.data = (int*)malloc(sizeof(int) * m);
                memcpy(feature_info->findex.data, findex, sizeof(int) * m);
                feature_info->findex.row = m;
                feature_info->findex.column = 1;
                feature_info->findex.row_skip = 0;
                feature_info->findex.skip4 = 0;
            }
            

            // solve the LCP problem and get lambda.
            // this will destroy A but that's OK
#if DebugPrint
            {
                int mlocal = dPAD(m);
                unsigned int mskip = dPAD(mlocal);
                dReal* Abk = memarena->AllocateArray<dReal>(mlocal * (size_t)mskip);
                memcpy(Abk, A, sizeof(dReal) * mlocal * (size_t)mskip);
                dReal* rhsbk = memarena->AllocateArray<dReal>(mlocal);
                memcpy(rhsbk, rhs, sizeof(dReal) * mlocal);

                PrintMat("rhs", rhs, m, 1, false, 1);
                dSolveLCP(memarena, m, A, lambda, rhs, feature_info->lcp_w.data, nub, lo, hi, findex); // Modify by Zhenhua Song: convert `outer_w` from NULL to feature_info->lcp_w.data
                PrintMat("x", lambda, m, 1, false, 1);

                printf("rhserr:\n");
                for (unsigned int i = 0; i < m; ++i)
                {
                    for (unsigned int j = 0; j < m; ++j)
                    {
                        if (j <= i)
                            rhsbk[i] -= Abk[i * mskip + j] * lambda[j];
                        else
                            rhsbk[i] -= Abk[j * mskip + i] * lambda[j];
                    }
                    printf("%0.10f\n", rhsbk[i]);
                }
                printf("\n");
            }
#else
      // (J M^{-1} J^T + 1/h CFM) \lambda = c/h - J * (v/h + invM * fe)
            dSolveLCP(memarena, m, A, lambda, rhs, feature_info->lcp_w.data, nub, lo, hi, findex); // Modify by Zhenhua Song
#endif
            {
                feature_info->lcp_lambda.data = (dReal*)malloc(sizeof(dReal) * m); // DO NOT forget to memory release..
                memcpy(feature_info->lcp_lambda.data, lambda, sizeof(dReal) * m);
                feature_info->lcp_lambda.column = 1;
                feature_info->lcp_lambda.row = m;
                feature_info->lcp_lambda.row_skip = 0;
                feature_info->lcp_lambda.skip4 = 0;

                feature_info->lcp_w.column = 1;
                feature_info->lcp_w.row = m;
                feature_info->lcp_w.row_skip = 0;
                feature_info->lcp_w.skip4 = 0;
            }
            
        } END_STATE_SAVE(memarena, lcpstate);

        {
            IFTIMING(dTimerNow("compute constraint force"));

            // compute the constraint force `cforce'
            // compute cforce = J'*lambda
            unsigned ofsi = 0;
            const dJointWithInfo1* jicurr = jointiinfos;
            const dJointWithInfo1* const jiend = jicurr + nj;
            for (; jicurr != jiend; ++jicurr) {
                const unsigned int infom = jicurr->info.m;
                dxJoint* joint = jicurr->joint;

                const dReal* JJ = J + 2 * 8 * (size_t)ofsi;
                const dReal* lambdarow = lambda + ofsi;

                dJointFeedback* fb = joint->feedback;

                if (fb) {
                    // the user has requested feedback on the amount of force that this
                    // joint is applying to the bodies. we use a slightly slower
                    // computation that splits out the force components and puts them
                    // in the feedback structure.
                    dReal data[8];
                    Multiply1_8q1(data, JJ, lambdarow, infom);

                    dxBody* b1 = joint->node[0].body;
                    dReal* cf1 = cforce + 8 * (size_t)(unsigned)b1->tag;
                    cf1[0] += (fb->f1[0] = data[0]);
                    cf1[1] += (fb->f1[1] = data[1]);
                    cf1[2] += (fb->f1[2] = data[2]);
                    cf1[4] += (fb->t1[0] = data[4]);
                    cf1[5] += (fb->t1[1] = data[5]);
                    cf1[6] += (fb->t1[2] = data[6]);

                    dxBody* b2 = joint->node[1].body;
                    if (b2) {
                        Multiply1_8q1(data, JJ + 8 * (size_t)infom, lambdarow, infom);

                        dReal* cf2 = cforce + 8 * (size_t)(unsigned)b2->tag;
                        cf2[0] += (fb->f2[0] = data[0]);
                        cf2[1] += (fb->f2[1] = data[1]);
                        cf2[2] += (fb->f2[2] = data[2]);
                        cf2[4] += (fb->t2[0] = data[4]);
                        cf2[5] += (fb->t2[1] = data[5]);
                        cf2[6] += (fb->t2[2] = data[6]);
                    }
                }
                else {
                    // no feedback is required, let's compute cforce the faster way
                    dxBody* b1 = joint->node[0].body;
                    dReal* cf1 = cforce + 8 * (size_t)(unsigned)b1->tag;
                    MultiplyAdd1_8q1(cf1, JJ, lambdarow, infom);

                    dxBody* b2 = joint->node[1].body;
                    if (b2) {
                        dReal* cf2 = cforce + 8 * (size_t)(unsigned)b2->tag;
                        MultiplyAdd1_8q1(cf2, JJ + 8 * (size_t)infom, lambdarow, infom);
                    }
                }

                ofsi += infom;
            }
        }
    } // if (m > 0)

    {
        // compute the velocity update
        IFTIMING(dTimerNow("compute velocity update"));

        // add fe to cforce and multiply cforce by stepsize
        dReal data[4];
        const dReal* invIrow = invI;
        dReal* cforcecurr = cforce;
        dxBody* const* const bodyend = body + nb;
        for (dxBody* const* bodycurr = body; bodycurr != bodyend; invIrow += 12, cforcecurr += 8, ++bodycurr) {
            dxBody* b = *bodycurr;

            dReal body_invMass_mul_stepsize = stepsize * b->invMass;
            for (unsigned int j = 0; j < 3; ++j) b->lvel[j] += (cforcecurr[j] + b->facc[j]) * body_invMass_mul_stepsize;

            for (unsigned int k = 0; k < 3; ++k) data[k] = (cforcecurr[4 + k] + b->tacc[k]) * stepsize;
            dMultiplyAdd0_331(b->avel, invIrow, data); // I^{-1} * (cforce + tau) * h = I^{-1} * cforce + \omega_{t}
        }
    }

    {
        // update the position and orientation from the next linear/angular velocity
        // (over the given timestep)
        IFTIMING(dTimerNow("update position"));
        dxBody* const* const bodyend = body + nb;
        for (dxBody* const* bodycurr = body; bodycurr != bodyend; ++bodycurr) {
            dxBody* b = *bodycurr;
            dxStepBody(b, stepsize);
        }
    }

    {
        IFTIMING(dTimerNow("tidy up"));

        // zero all force accumulators
        dxBody* const* const bodyend = body + nb;
        for (dxBody* const* bodycurr = body; bodycurr != bodyend; ++bodycurr) {
            dxBody* b = *bodycurr;
            b->facc[0] = 0;
            b->facc[1] = 0;
            b->facc[2] = 0;
            b->facc[3] = 0;
            b->tacc[0] = 0;
            b->tacc[1] = 0;
            b->tacc[2] = 0;
            b->tacc[3] = 0;
        }
    }

    IFTIMING(dTimerEnd());
    if (m > 0) IFTIMING(dTimerReport(stdout, 1));

}

//****************************************************************************

void dInternalStepIslandWithInfo(dxWorldProcessMemArena* memarena,
    dxWorld* world, dxBody* const* body, unsigned int nb,
    dxJoint* const* joint, unsigned int nj, dReal stepsize, WorldStepFeatureInfoPtr feature_info)
{
    dInternalStepIslandWithInfo_x2(memarena, world, body, nb, joint, nj, stepsize, feature_info);
}

