/*************************************************************************

BSD 3-Clause License

Copyright (c) 2023,  Visual Computing and Learning Lab, Peking University

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*************************************************************************/


#include <ode/dampedstepcommon.h>
#include <ode/extutils.h>
#include "step.h"

static void dInternalDamppedStepIslandWithInfo_x2(dxWorldProcessMemArena* memarena,
    dxWorld* world, dxBody* const* body, unsigned int nb,
    dxJoint* const* _joint, unsigned int _nj, dReal stepsize, WorldStepFeatureInfoPtr feature_info)
{
    IFTIMING(dTimerStart("preprocessing"));

    const dReal stepsizeRecip = dRecip(stepsize);

    {
        // number all bodies in the body list - set their tag values
        unsigned int i;
        for (i = 0; i < nb; ++i) body[i]->tag = i;
    }

    ///////////////////////////////////////////////////////
    // added by Libin Liu
    // compute damping matrix
    // for a joint between p(k) and k
    //  damping_k    = K (w_k - w_p(k))
    //  damping_p(k) = K (w_p(k) - w_k)
    // actually, we don't really use the damping matrix D
    // we will only use the inverse of (M + D);
    ///////////////////////////////////////////////////////
    // compute D matrix and constuct the structure of the sparse matrix
    const unsigned int iplusdBlockDim = 3;
    unsigned int iplusdDim = nb * iplusdBlockDim;
    unsigned int iplusdSkip = dPAD(iplusdDim);
    unsigned int iplusdSize = iplusdDim * iplusdSkip;
    dReal* iplusd = memarena->AllocateArray<dReal>((size_t)iplusdSize);
    dSetZero(iplusd, iplusdSize);
    dReal* iplusdinv = NULL;

    BEGIN_STATE_SAVE(memarena, dampingstate)
    {
        char* iplusdMask = memarena->AllocateArray<char>((size_t)iplusdSize);
        memset(iplusdMask, 0, iplusdSize * sizeof(char));

        dxJoint* const* const _jend = _joint + _nj;
        dxJoint* const* _jcurr = _joint;
        dReal* block = NULL;
        char* maskBlock = NULL;
        for (; _jcurr != _jend; ++_jcurr)
        {
            dxJoint* joint = *_jcurr;

            //joint->isAnisotropicDamping = false;
            //joint->aveldamping[0] = 0;
            if (joint->isAnisotropicDamping && joint->dampingRefBody)
            {
                dMatrix3 D = {
                    joint->aveldamping[0] * stepsize, 0, 0, 0,
                    0, joint->aveldamping[1] * stepsize, 0, 0,
                    0, 0, joint->aveldamping[2] * stepsize, 0
                };
                // the damping matrix is defined in ref frame
                // we should convert it to global frame
                dMatrix3 tmp;
                dMultiply2_333(tmp, D, joint->dampingRefBody->posr.R);
                dMultiply0_333(D, joint->dampingRefBody->posr.R, tmp);

                int bid0 = joint->node[0].body->tag;
                int blockId0 = ((bid0 * iplusdSkip + bid0) * iplusdBlockDim);
                block = iplusd + blockId0;
                block[0] += D[0]; block[1] += D[1]; block[2] += D[2];
                block[iplusdSkip] += D[4]; block[iplusdSkip + 1] += D[5]; block[iplusdSkip + 2] += D[6];
                block[iplusdSkip * 2] += D[8]; block[iplusdSkip * 2 + 1] += D[9]; block[iplusdSkip * 2 + 2] += D[10];

                maskBlock = iplusdMask + blockId0;
                maskBlock[0] = 1; maskBlock[1] = 1; maskBlock[2] = 1;
                maskBlock[iplusdSkip] = 1; maskBlock[iplusdSkip + 1] = 1; maskBlock[iplusdSkip + 2] = 1;
                maskBlock[iplusdSkip * 2] = 1; maskBlock[iplusdSkip * 2 + 1] = 1; maskBlock[iplusdSkip * 2 + 2] = 1;

                if (joint->node[1].body)
                {
                    int bid1 = joint->node[1].body->tag;
                    int blockId1 = ((bid1 * iplusdSkip + bid1) * iplusdBlockDim);
                    block = iplusd + blockId1;
                    block[0] += D[0]; block[1] += D[1]; block[2] += D[2];
                    block[iplusdSkip] += D[4]; block[iplusdSkip + 1] += D[5]; block[iplusdSkip + 2] += D[6];
                    block[iplusdSkip * 2] += D[8]; block[iplusdSkip * 2 + 1] += D[9]; block[iplusdSkip * 2 + 2] += D[10];

                    maskBlock = iplusdMask + blockId1;
                    maskBlock[0] = 1; maskBlock[1] = 1; maskBlock[2] = 1;
                    maskBlock[iplusdSkip] = 1; maskBlock[iplusdSkip + 1] = 1; maskBlock[iplusdSkip + 2] = 1;
                    maskBlock[iplusdSkip * 2] = 1; maskBlock[iplusdSkip * 2 + 1] = 1; maskBlock[iplusdSkip * 2 + 2] = 1;

                    int blockId12 = 0;
                    int blockId21 = 0;
                    if (bid0 > bid1)
                    {
                        blockId12 = ((bid0 * iplusdSkip + bid1) * iplusdBlockDim);
                        //blockId21 = ((bid1 * iplusdSkip + bid0) * iplusdBlockDim);
                    }
                    else
                    {
                        blockId12 = ((bid1 * iplusdSkip + bid0) * iplusdBlockDim);
                        //blockId21 = ((bid0 * iplusdSkip + bid1) * iplusdBlockDim); 
                    }

                    block = iplusd + blockId12;
                    block[0] -= D[0]; block[1] -= D[1]; block[2] -= D[2];
                    block[iplusdSkip] -= D[4]; block[iplusdSkip + 1] -= D[5]; block[iplusdSkip + 2] -= D[6];
                    block[iplusdSkip * 2] -= D[8]; block[iplusdSkip * 2 + 1] -= D[9]; block[iplusdSkip * 2 + 2] -= D[10];

                    //block = iplusd + blockId21;
                    //block[             0] -= D[0]; block[                 1] -= D[1]; block[                 2] -= D[2];
                    //block[iplusdSkip    ] -= D[4]; block[iplusdSkip     + 1] -= D[5]; block[iplusdSkip     + 2] -= D[6];
                    //block[iplusdSkip * 2] -= D[8]; block[iplusdSkip * 2 + 1] -= D[9]; block[iplusdSkip * 2 + 2] -= D[10];

                    maskBlock = iplusdMask + blockId12;
                    maskBlock[0] = 1; maskBlock[1] = 1; maskBlock[2] = 1;
                    maskBlock[iplusdSkip] = 1; maskBlock[iplusdSkip + 1] = 1; maskBlock[iplusdSkip + 2] = 1;
                    maskBlock[iplusdSkip * 2] = 1; maskBlock[iplusdSkip * 2 + 1] = 1; maskBlock[iplusdSkip * 2 + 2] = 1;

                    //maskBlock = iplusdMask + blockId21;
                    //maskBlock[             0] = 1; maskBlock[                 1] = 1; maskBlock[                 2] = 1;
                    //maskBlock[iplusdSkip    ] = 1; maskBlock[iplusdSkip     + 1] = 1; maskBlock[iplusdSkip     + 2] = 1;
                    //maskBlock[iplusdSkip * 2] = 1; maskBlock[iplusdSkip * 2 + 1] = 1; maskBlock[iplusdSkip * 2 + 2] = 1;

                }
            }
            else
            {
                double d = joint->aveldamping[0] * stepsize;
                int bid0 = joint->node[0].body->tag;
                int blockId0 = ((bid0 * iplusdSkip + bid0) * iplusdBlockDim);
                block = iplusd + blockId0;
                block[0] += d; block[iplusdSkip + 1] += d; block[iplusdSkip * 2 + 2] += d;

                maskBlock = iplusdMask + blockId0;
                maskBlock[0] = maskBlock[iplusdSkip + 1] = maskBlock[iplusdSkip * 2 + 2] = 1;

                if (joint->node[1].body)
                {
                    int bid1 = joint->node[1].body->tag;
                    int blockId1 = ((bid1 * iplusdSkip + bid1) * iplusdBlockDim);
                    block = iplusd + blockId1;
                    block[0] += d; block[iplusdSkip + 1] += d; block[iplusdSkip * 2 + 2] += d;

                    maskBlock = iplusdMask + blockId1;
                    maskBlock[0] = maskBlock[iplusdSkip + 1] = maskBlock[iplusdSkip * 2 + 2] = 1;

                    int blockId12 = 0;
                    //int blockId21 = 0;
                    if (bid0 > bid1)
                    {
                        blockId12 = ((bid0 * iplusdSkip + bid1) * iplusdBlockDim);
                        //blockId21 = ((bid1 * iplusdSkip + bid0) * iplusdBlockDim);
                    }
                    else
                    {
                        blockId12 = ((bid1 * iplusdSkip + bid0) * iplusdBlockDim);
                        //blockId21 = ((bid0 * iplusdSkip + bid1) * iplusdBlockDim); 
                    }


                    block = iplusd + blockId12;
                    block[0] -= d; block[iplusdSkip + 1] -= d; block[iplusdSkip * 2 + 2] -= d;

                    //block = iplusd + blockId21;
                    //block[0] -= d; block[iplusdSkip + 1] -= d; block[iplusdSkip * 2 + 2] -= d;

                    maskBlock = iplusdMask + blockId12;
                    maskBlock[0] = maskBlock[iplusdSkip + 1] = maskBlock[iplusdSkip * 2 + 2] = 1;

                    //maskBlock = iplusdMask + blockId21;
                    //maskBlock[0] = maskBlock[iplusdSkip + 1] = maskBlock[iplusdSkip * 2 + 2] = 1;
                }
            }
        }


        ///////////////////////////////////////////////////////


        // for all bodies, compute the inertia tensor and its inverse in the global
        // frame, and compute the rotational force and add it to the torque
        // accumulator. invI are vertically stacked 3x4 matrices, one per body.
        // @@@ check computation of rotational force.

        ////////////////////////////////
        // added by Libin
        // inverse inertia tensor is no longer necessary, since we will compute (I+D)^-1
        //dReal *invI = memarena->AllocateArray<dReal> (3*4*(size_t)nb);

         // Identical to QuickStep
          //dReal *invIrow = invI;
        dxBody* const* const bodyend = body + nb;
        for (dxBody* const* bodycurr = body; bodycurr != bodyend; /*invIrow += 12, */++bodycurr) {
            dMatrix3 tmp;
            dxBody* b = *bodycurr;

            ////////////////////////////////
            // added by Libin
            // inverse inertia tensor is no longer necessary, since we will compute (M+D)^-1
            // compute inverse inertia tensor in global frame
            //dMultiply2_333 (tmp,b->invI,b->posr.R);
            //dMultiply0_333 (invIrow,b->posr.R,tmp);

            // compute inertia tensor in global frame
            dMultiply2_333(tmp, b->mass.I, b->posr.R);
            dMultiply0_333(b->curI, b->posr.R, tmp);
            // compute I+D
            int bid = b->tag;
            int blockId = ((bid * iplusdSkip + bid) * iplusdBlockDim);
            block = iplusd + blockId;
            block[0] += b->curI[0]; block[1] += b->curI[1]; block[2] += b->curI[2];
            block[iplusdSkip] += b->curI[4]; block[iplusdSkip + 1] += b->curI[5]; block[iplusdSkip + 2] += b->curI[6];
            block[iplusdSkip * 2] += b->curI[8]; block[iplusdSkip * 2 + 1] += b->curI[9]; block[iplusdSkip * 2 + 2] += b->curI[10];

            maskBlock = iplusdMask + blockId;
            maskBlock[0] = 1; maskBlock[1] = 1; maskBlock[2] = 1;
            maskBlock[iplusdSkip] = 1; maskBlock[iplusdSkip + 1] = 1; maskBlock[iplusdSkip + 2] = 1;
            maskBlock[iplusdSkip * 2] = 1; maskBlock[iplusdSkip * 2 + 1] = 1; maskBlock[iplusdSkip * 2 + 2] = 1;

            // compute Iw
            dMultiply0_331(tmp, b->curI, b->avel);

            if (b->flags & dxBodyGyroscopic) {
                // compute rotational force
                dSubtractVectorCross3(b->tacc, b->avel, tmp);
            }

            dScaleVector3(tmp, stepsizeRecip);
            dAddVectors3(b->tacc, b->tacc, tmp);

            //double mass_time = b->mass.mass * stepsizeRecip;
            //b->facc[0] += mass_time * b->lvel[0];
            //b->facc[1] += mass_time * b->lvel[1];
            //b->facc[2] += mass_time * b->lvel[2];
        }

#if DebugPrint
        //for (unsigned int i = 0; i < iplusdDim; ++i)
        //{
        //    for (unsigned int j = 0; j < iplusdDim; ++j)
        //        printf("%9.20f ", iplusd[i * iplusdSkip + j]);
        //    printf("\n");
        //}
        PrintMat("I_h+D", iplusd, iplusdDim, iplusdDim, false, iplusdSkip);
        for (unsigned int i = 0; i < iplusdDim; ++i)
        {
            for (unsigned int j = 0; j < iplusdDim; ++j)
                printf("%d", iplusdMask[i * iplusdSkip + j]);
            printf("\n");
        }
#endif

        iplusdinv = iplusd;
        //unsigned int *pi = memarena->AllocateArray<unsigned int>(iplusdDim);
        //int rcmreq = RCM(iplusdMask, iplusdDim, iplusdSkip, pi, NULL);
        //char *rcmMem = memarena->AllocateArray<char>(rcmreq);
        //RCM(iplusdMask, iplusdDim, iplusdSkip, pi, rcmMem);

#if DebugPrint
    //for (unsigned int i = 0; i < iplusdDim; ++i)
    //    printf("%d ", pi[i]);
    //printf("\n");
    //
    //printf("-----\n");
    //for (unsigned int i = 0; i < iplusdDim; ++i)
    //{
    //    for (unsigned int j = 0; j < iplusdDim; ++j)
    //        printf("%d", iplusdMask[pi[i] * iplusdSkip + pi[j]]);
    //    printf("\n");
    //}
#endif

        int memreq = sparseInverse(iplusd, iplusdMask, iplusdDim, iplusdSkip, NULL, iplusdinv, iplusdSkip, NULL);
        char* tempMem = memarena->AllocateArray<char>(memreq);
        //memset(tempMem, 0, memreq);

        //dReal *iplusdinv = memarena->AllocateArray<dReal> ((size_t)iplusdSize);
        //memcpy(iplusdinv, iplusd, sizeof(dReal) * iplusdSize);
        //std::swap(iplusd, iplusdinv);

        //sparseInverse(iplusd, iplusdMask, iplusdDim, iplusdSkip, pi, iplusdinv, iplusdSkip, tempMem);
        sparseInverse(iplusd, iplusdMask, iplusdDim, iplusdSkip, iplusdinv, iplusdSkip, tempMem);

#if DebugPrint
        //for (unsigned int i = 0; i < iplusdDim; ++i)
        //{
        //    for (unsigned int j = 0; j < iplusdDim; ++j)
        //        printf("%10.3g ", iplusdinv[i * iplusdSkip + j]);
        //    printf("\n");
        //}
        PrintMat("invID", iplusdinv, iplusdDim, iplusdDim, false, iplusdSkip);
        double e = 0;
        double se = 0;
        for (unsigned i = 0; i < iplusdDim; ++i)
        {
            for (unsigned j = 0; j < iplusdDim; ++j)
            {
                double rij = 0;
                for (unsigned k = 0; k < iplusdDim; ++k)
                    rij += iplusd[i > k ? (i * iplusdSkip + k) : (k * iplusdSkip + i)] * iplusdinv[k * iplusdSkip + j];
                if (i == j)
                    rij -= 1.0;
                e += abs(rij);

                se += abs(iplusd[i * iplusdSkip + j] - iplusd[j * iplusdSkip + i]);
            }
        }
        printf("inverr: %0.20f\n", e);
        printf("symerr: %0.20f\n", se);
#endif

        iplusd = NULL;
    }
    END_STATE_SAVE(memarena, dampingstate);

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

#if DebugPrint
    {
        const dxBody* const* const bodyend = body + nb;
        printf("invMh lvel avel facc tacc\n");
        for (const dxBody* const* bodycurr = body; bodycurr != bodyend; ++bodycurr)
        {
            const dxBody* b = *bodycurr;
            //printf(" %12.9f", invMh[bodycurr - body]);
            double mass_time = b->mass.mass * stepsizeRecip;
            printf(" %12.9f %12.9f %12.9f", b->lvel[0], b->lvel[1], b->lvel[2]);
            printf(" %12.9f %12.9f %12.9f", b->avel[0], b->avel[1], b->avel[2]);
            printf(" %12.9f %12.9f %12.9f",
                b->facc[0] + mass_time * b->lvel[0],
                b->facc[1] + mass_time * b->lvel[1],
                b->facc[2] + mass_time * b->lvel[2]);
            printf(" %12.9f %12.9f %12.9f", b->tacc[0], b->tacc[1], b->tacc[2]);
            printf("\n");
        }
    }
#endif
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
                    dIASSERT(jicurr->info.m >= 0 /*&& jicurr->info.m <= 6*/ && jicurr->info.nub >= 0 && jicurr->info.nub <= jicurr->info.m);
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

            dReal* Jinvm = memarena->AllocateArray<dReal>(2 * 4 * (size_t)m);
            dSetZero(Jinvm, 2 * 4 * (size_t)m);

            unsigned int nid = nb * 4;
            dReal* JinvID = memarena->AllocateArray<dReal>(nb * 4 * (size_t)m);
            dSetZero(JinvID, nid * (size_t)m);

            {
                IFTIMING(dTimerNow("create J"));
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
            }

            {
                IFTIMING(dTimerNow("compute A"));
                {
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
                        //dReal *body_invI0 = invI + (size_t)b0*12;
                        dReal* Jsrc = J + 2 * 8 * (size_t)ofsi;
                        dReal* Jdstm = Jinvm + 2 * 4 * (size_t)ofsi;
                        dReal* JdstID = JinvID + (size_t)(nid * ofsi);
                        dReal* iplusdinv0 = iplusdinv + (size_t)(iplusdSkip * b0 * iplusdBlockDim);

                        dReal* JdstID0 = JdstID;
                        for (unsigned int j = infom; j > 0;) {
                            j -= 1;
                            for (unsigned int k = 0; k < 3; ++k) Jdstm[k] = Jsrc[k] * body_invMass0;
                            Jsrc += 4;
                            Jdstm += 4;

                            dReal* iplusdinv00 = iplusdinv0;
                            dReal* iplusdinv01 = iplusdinv00 + iplusdSkip;
                            dReal* iplusdinv02 = iplusdinv01 + iplusdSkip;

                            for (unsigned int b = 0; b < nb; ++b)
                            {
                                for (unsigned k = 0; k < 3; ++k)
                                {
                                    JdstID0[k] += Jsrc[0] * iplusdinv00[k]
                                        + Jsrc[1] * iplusdinv01[k]
                                        + Jsrc[2] * iplusdinv02[k];
                                }
                                JdstID0 += 4;
                                iplusdinv00 += 3;
                                iplusdinv01 += 3;
                                iplusdinv02 += 3;
                            }
                            //dMultiply0_133 (JinvID,Jsrc+4,body_invI0);
                            Jsrc += 4;
                        }

                        if (joint->node[1].body) {
                            unsigned int b1 = joint->node[1].body->tag;
                            dReal body_invMass1 = body[b1]->invMass;
                            //dReal *body_invI1 = invI + (size_t)b1*12;

                            dReal* iplusdinv1 = iplusdinv + (size_t)(iplusdSkip * b1 * iplusdBlockDim);
                            dReal* JdstID1 = JdstID;
                            for (unsigned int j = infom; j > 0;) {
                                j -= 1;
                                for (unsigned int k = 0; k < 3; ++k) Jdstm[k] = Jsrc[k] * body_invMass1;
                                Jsrc += 4;
                                Jdstm += 4;

                                dReal* iplusdinv10 = iplusdinv1;
                                dReal* iplusdinv11 = iplusdinv10 + iplusdSkip;
                                dReal* iplusdinv12 = iplusdinv11 + iplusdSkip;
                                for (unsigned int b = 0; b < nb; ++b)
                                {
                                    for (unsigned k = 0; k < 3; ++k)
                                    {
                                        JdstID1[k] += Jsrc[0] * iplusdinv10[k]
                                            + Jsrc[1] * iplusdinv11[k]
                                            + Jsrc[2] * iplusdinv12[k];
                                    }
                                    JdstID1 += 4;
                                    iplusdinv10 += 3;
                                    iplusdinv11 += 3;
                                    iplusdinv12 += 3;
                                }
                                //dMultiply0_133 (Jdst+4,Jsrc+4,body_invI1);
                                Jsrc += 4;
                            }
                        }

                        ofsi += infom;
                    }
                }

#if DebugPrint
                // print J
                {
                    unsigned int mlocal = dPAD(m);
                    dReal* JCur = J;
                    printf("J: \n");
                    for (unsigned int i = 0; i < m * 2; ++i, JCur += 8)
                    {
                        printf("%0.20f %0.20f %0.20f %0.20f %0.20f %0.20f\n",
                            JCur[0], JCur[1], JCur[2], JCur[4], JCur[5], JCur[6]);
                    }
                }
#endif

                {
                    // now compute A = JinvM * J'. A's rows and columns are grouped by joint,
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
                            dReal* Jinvmrow = Jinvm + 2 * 4 * (size_t)ofsi;

                            dxBody* jb0 = joint->node[0].body;
                            for (dxJointNode* n0 = jb0->firstjoint; n0; n0 = n0->next) {
                                // if joint was tagged as -1 then it is an inactive (m=0 or disabled)
                                // joint that should not be considered
                                int j0 = n0->joint->tag;
                                if (j0 != -1 && (unsigned)j0 < i) {
                                    const dJointWithInfo1* jiother = jointiinfos + j0;
                                    size_t ofsother = (jiother->joint->node[1].body == jb0) ? 8 * (size_t)jiother->info.m : 0;
                                    // set block of A
                                    MultiplyAdd2_p4r(Arow + ofs[j0], Jinvmrow,
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
                                        MultiplyAdd2_p4r(Arow + ofs[j1], Jinvmrow + 4 * (size_t)infom,
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
                        dReal* Jinvmrow = Jinvm + 2 * 4 * (size_t)ofsi;
                        dReal* Jrow = J + 2 * 8 * (size_t)ofsi;
                        Multiply2_p4r(Arow, Jinvmrow, Jrow, infom, infom, mskip);
                        if (jicurr->joint->node[1].body) {
                            MultiplyAdd2_p4r(Arow, Jinvmrow + 4 * (size_t)infom, Jrow + 8 * (size_t)infom, infom, infom, mskip);
                        }

                        ofsi += infom;
                    }
                }

                {
                    // add the inertia-damping part to A
                    const unsigned int mskip = dPAD(m);

                    // JinvID = J * inv(I+D)
                    unsigned ofsi = 0;
                    const dJointWithInfo1* const jiend = jointiinfos + nj;
                    const dJointWithInfo1* jicurri = jointiinfos;
                    for (; jicurri != jiend; ++jicurri) {
                        const unsigned int infomi = jicurri->info.m;
                        dReal* Arow = A + mskip * (size_t)ofsi;
                        dReal* JinvIDRow = JinvID + (size_t)(nid * ofsi);

                        unsigned ofsj = 0;
                        const dJointWithInfo1* jicurrj = jointiinfos;
                        for (; jicurrj != jiend; ++jicurrj)
                        {
                            dReal* Jrow = J + 2 * 8 * (size_t)ofsj + 4;

                            const unsigned int infomj = jicurrj->info.m;
                            dxJoint* joint = jicurrj->joint;
                            unsigned int b0 = joint->node[0].body->tag;
                            dReal* JinvIDBlock0 = JinvIDRow + (b0 * 4);
                            MultiplyAdd2_p4r(Arow + (size_t)ofsj, JinvIDBlock0, Jrow, infomi, infomj, mskip, nid, 8);

                            if (joint->node[1].body) {
                                unsigned int b1 = joint->node[1].body->tag;
                                dReal* JinvIDBlock1 = JinvIDRow + (b1 * 4);
                                MultiplyAdd2_p4r(Arow + (size_t)ofsj, JinvIDBlock1, Jrow + 8 * (size_t)infomj, infomi, infomj, mskip, nid, 8);
                            }

                            ofsj += infomj;
                        }

                        ofsi += infomi;
                    }
                }


                {
                    // add cfm to the diagonal of A
                    const unsigned int mskip = dPAD(m);

                    dReal* Arow = A;
                    for (unsigned int i = 0; i < m; Arow += mskip, ++i) {
                        Arow[i] += cfm[i] * stepsizeRecip;
                    }

#if DebugPrint
                    PrintMat("A", A, m, m, false, mskip);

                    printf("A:\n");
                    for (unsigned int i = 0; i < m; ++i)
                    {
                        for (unsigned int j = 0; j < m; ++j)
                            printf("%14.10f ", A[i * mskip + j]);
                        printf("\n");
                    }
#endif

                }
            }

        } END_STATE_SAVE(memarena, cfmstate);


#if DebugPrint
        for (unsigned int i = 0; i < iplusdDim; ++i)
        {
            for (unsigned int j = 0; j < iplusdDim; ++j)
                printf("%9.4f ", iplusdinv[i * iplusdSkip + j]);
            printf("\n");
        }
#endif

#if DebugPrint
        {
            printf("tacc\n");
            dxBody* const* const bodyend = body + nb;
            for (dxBody* const* bodycurri = body; bodycurri != bodyend; ++bodycurri) {
                dxBody* bi = *bodycurri;
                printf("%9.4f %9.4f %9.4f ", bi->tacc[0], bi->tacc[1], bi->tacc[2]);
            }
            printf("\n");
        }
#endif

        BEGIN_STATE_SAVE(memarena, tmp1state) {
            // compute the right hand side `rhs'
            IFTIMING(dTimerNow("compute rhs"));

            dReal* tmp1 = memarena->AllocateArray<dReal>((size_t)nb * 8);
            dSetZero(tmp1, nb * 8);

            {
                // put (M*invID)v/h + invID*fe into tmp1
                dReal* tmp1curr = tmp1;
                dxBody* const* const bodyend = body + nb;
                for (dxBody* const* bodycurri = body; bodycurri != bodyend; tmp1curr += 8, ++bodycurri) {
                    dxBody* bi = *bodycurri;
                    for (unsigned int j = 0; j < 3; ++j) tmp1curr[j] = bi->facc[j] * bi->invMass + bi->lvel[j] * stepsizeRecip;

                    const dReal* invIDrow = iplusdinv + (size_t)(iplusdSkip * bi->tag * iplusdBlockDim);
                    for (dxBody* const* bodycurrj = body; bodycurrj != bodyend; ++bodycurrj)
                    {
                        dxBody* bj = *bodycurrj;
                        const dReal* invIDBlockRow0 = invIDrow + (size_t)(bj->tag * iplusdBlockDim);
                        const dReal* invIDBlockRow1 = invIDBlockRow0 + (size_t)iplusdSkip;
                        const dReal* invIDBlockRow2 = invIDBlockRow1 + (size_t)iplusdSkip;
                        //dMultiply0_331 (tmp1curr+4, invIrow, b->tacc);
                        double res0 = 0, res1 = 0, res2 = 0;
                        for (int k = 0; k < 3; ++k)
                            res0 += invIDBlockRow0[k] * bj->tacc[k];
                        for (int k = 0; k < 3; ++k)
                            res1 += invIDBlockRow1[k] * bj->tacc[k];
                        for (int k = 0; k < 3; ++k)
                            res2 += invIDBlockRow2[k] * bj->tacc[k];
                        tmp1curr[4] += res0;
                        tmp1curr[5] += res1;
                        tmp1curr[6] += res2;
                    }
                }
            }

            {
                // init rhs -- this erases 'c' as they reside in the same memory!!!
                rhs = c;
                for (unsigned int i = 0; i < m; ++i) rhs[i] = c[i] * stepsizeRecip;
                c = NULL; // set 'c' to NULL to prevent unexpected access
            }

            {
                // put J*tmp1 into rhs
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

        dReal* lambda0 = memarena->AllocateArray<dReal>(m);

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
                dSolveLCP(memarena, m, A, lambda0, rhs, feature_info->lcp_w.data, nub, lo, hi, findex);
                printf("rhserr:\n");
                for (unsigned int i = 0; i < m; ++i)
                {
                    for (unsigned int j = 0; j < m; ++j)
                    {
                        if (i >= j)
                            rhsbk[i] -= Abk[i * mskip + j] * lambda0[j];
                        else
                            rhsbk[i] -= Abk[j * mskip + i] * lambda0[j];
                    }
                    printf("  %0.10f\n", rhsbk[i]);
                }
                printf("\n");
            }
#else
            dSolveLCP(memarena, m, A, lambda0, rhs, feature_info->lcp_w.data, nub, lo, hi, findex);
#endif
            {
                feature_info->lcp_lambda.data = (dReal*)malloc(sizeof(dReal) * m); // DO NOT forget to memory release..
                memcpy(feature_info->lcp_lambda.data, lambda0, sizeof(dReal) * m);
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
                const dReal* lambdarow = lambda0 + ofsi;

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
#if DebugPrint
        {
            printf("cforce:\n");
            dReal* cforcecurr = cforce;
            dxBody* const* bodycurr = body;
            for (unsigned int i = 0; i < nb; ++i, cforcecurr += 8, ++bodycurr)
            {
                for (unsigned int j = 0; j < 3; ++j)
                    printf("%9.6g ", cforcecurr[j]);
                double t = 0;
                for (unsigned int j = 0; j < 3; ++j)
                {
                    printf("%9.6g ", cforcecurr[4 + j]);
                    t += cforcecurr[4 + j] * (*bodycurr)->avel[j];
                }

                printf("%9.6g\n", t);
            }
            printf("\n");
        }
#endif
        // add fe to cforce and multiply cforce by stepsize
        //const dReal *invIrow = invI;
        dReal* cforcecurr = cforce;
        dxBody* const* const bodyend = body + nb;
        // update linear velocity
        for (dxBody* const* bodycurri = body; bodycurri != bodyend; cforcecurr += 8, ++bodycurri) {
            dxBody* bi = *bodycurri;

            dReal body_invMass_mul_stepsize = stepsize * bi->invMass;
            for (unsigned int j = 0; j < 3; ++j) bi->lvel[j] += (cforcecurr[j] + bi->facc[j]) * body_invMass_mul_stepsize;

            dAddVectors3(bi->tacc, bi->tacc, cforcecurr + 4);
        }

        const dReal* invIDrow = iplusdinv;
        for (dxBody* const* bodycurri = body; bodycurri != bodyend; ++bodycurri, invIDrow += iplusdSkip * iplusdBlockDim)
        {
            dxBody* bi = *bodycurri;
            dSetZero(bi->avel, 4);

            const dReal* invIDBlockRow0 = invIDrow;
            for (dxBody* const* bodycurrj = body; bodycurrj != bodyend; ++bodycurrj, invIDBlockRow0 += iplusdBlockDim)
            {
                dxBody* bj = *bodycurrj;
                const dReal* invIDBlockRow1 = invIDBlockRow0 + (size_t)iplusdSkip;
                const dReal* invIDBlockRow2 = invIDBlockRow1 + (size_t)iplusdSkip;
                //dMultiply0_331 (tmp1curr+4, invIrow, b->tacc);
                double res0 = 0, res1 = 0, res2 = 0;
                for (int k = 0; k < 3; ++k)
                    res0 += invIDBlockRow0[k] * bj->tacc[k];
                for (int k = 0; k < 3; ++k)
                    res1 += invIDBlockRow1[k] * bj->tacc[k];
                for (int k = 0; k < 3; ++k)
                    res2 += invIDBlockRow2[k] * bj->tacc[k];

                // clamp avel to prevent crash
                if (!::isfinite(res0) || res0 > 1e20 || res0 < -1e20)
                    res0 = 1e20;
                if (!::isfinite(res1) || res1 > 1e20 || res1 < -1e20)
                    res1 = 1e20;
                if (!::isfinite(res2) || res2 > 1e20 || res2 < -1e20)
                    res2 = 1e20;

                bi->avel[0] += res0;
                bi->avel[1] += res1;
                bi->avel[2] += res2;

            }

            dScaleVector3(bi->avel, stepsize);
        }
    }

    {
        // update the position and orientation from the new linear/angular velocity
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

void dInternalDamppedStepIslandWithInfo(dxWorldProcessMemArena* memarena,
    dxWorld* world, dxBody* const* body, unsigned int nb,
    dxJoint* const* joint, unsigned int nj, dReal stepsize, WorldStepFeatureInfoPtr feature_info)
{
    dInternalDamppedStepIslandWithInfo_x2(memarena, world, body, nb, joint, nj, stepsize, feature_info);
}

extern "C" int dWorldDampedStepWithInfo(dWorldID w, dReal stepsize, WorldStepFeatureInfoPtr feature_info)
{
    dUASSERT(w, "bad world argument");
    dUASSERT(stepsize > 0, "stepsize must be > 0");

    bool result = false;

    dxWorldProcessIslandsInfo islandsinfo;
    if (dxReallocateWorldProcessContext(w, islandsinfo, stepsize, &dxEstimateDamppedStepMemoryRequirements))
    {
        dxProcessIslandsWithInfo(w, islandsinfo, stepsize, &dInternalDamppedStepIslandWithInfo, feature_info);

        result = true;
    }

    dxCleanupWorldProcessContext(w);

    return result;
}
