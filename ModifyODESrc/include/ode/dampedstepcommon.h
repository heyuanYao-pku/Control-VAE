#pragma once
#include "dampedstep.h"

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

// sort
#include <algorithm>
#include <fstream>

#include <ode/extutils.h> // Add by Zhenhua Song

//****************************************************************************


#ifdef TIMING
#define IFTIMING(x) x
#else
#define IFTIMING(x) ((void)0)
#endif

#ifdef _DEBUG
#define DebugPrint 0
#else
#define DebugPrint 0
#endif


#ifndef In
#define In
#endif
#ifndef Out
#define Out
#endif
#ifndef InOut
#define InOut
#endif
#ifndef InOpt
#define InOpt
#endif

// sort by key
template <class Index, class Key>
inline void SortByWeight(
    Index * idstart, Index * idend, Key * k
)
{
    std::sort(idstart, idend, [&k](const Index& l, const Index& r) { return k[l] < k[r]; });
}

int RCM(
    In const char* mask, // the outline mask, 0-entry is zero
    In unsigned int n, // dim
    In unsigned int nskip, // rows and row skip
    Out unsigned int* p, // 
    In void* memarea   // pre-allocated memory space for internal usage, if this parameter is zero, the function will estimate a maximum usage
);

void solveLhc(
    In unsigned int n,
    In unsigned int heading,
    In const dReal* L,
    In const dReal* c,
    Out dReal* h
);

int sparseInverse(
    In const dReal* mat, // a sparse SPD matrix
    In const char* matMask, // the outline mask, 0-entry is zero
    In unsigned int n, // dim
    In unsigned int nskip, // rows and row skip
    InOpt unsigned int* p, // permutation matrix for better performance
    Out dReal* invmat, // nxn square matrix
    In unsigned int ninvskip,
    In void* memarea   // pre-allocated memory space for internal usage, if this parameter is zero, the function will estimate a maximum usage
);

int sparseInverse(
    In const dReal* mat, // a sparse SPD matrix
    In const char* matMask, // the outline mask, 0-entry is zero
    In unsigned int n, // dim
    In unsigned int nskip, // rows and row skip
    Out dReal* invmat, // nxn square matrix
    In unsigned int ninvskip,
    In void* memarea   // pre-allocated memory space for internal usage, if this parameter is zero, the function will estimate a maximum usage
);
