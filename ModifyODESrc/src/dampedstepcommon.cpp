#include <ode/dampedstepcommon.h>

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


// Reverse Cuthill-Mckee (RCM) algorithm
// The reverse Cuthill-McKee ordering is intended to reduce the profile 
// or bandwidth of the matrix. It is not guaranteed to find the smallest
// possible bandwidth, but it usually does. 

// NOTE: this implementation always take enough memory to ensure speed.
//   so that it is not very memory efficient

int RCM(
    In const char* mask, // the outline mask, 0-entry is zero
    In unsigned int n, // dim
    In unsigned int nskip, // rows and row skip
    Out unsigned int* p, // 
    In void* memarea   // pre-allocated memory space for internal usage, if this parameter is zero, the function will estimate a maximum usage
)
{
    if (!memarea)
    {
        int s = sizeof(unsigned int*) * n + sizeof(unsigned int) * (n * (n + 2))
            + sizeof(char) * n;
        return s;
    }

    // graph structure
    unsigned int** neighbors = (unsigned int**)memarea;
    memarea = (unsigned int**)memarea + n;
    unsigned int* degree = (unsigned int*)memarea;
    memarea = (unsigned int*)memarea + n;
    char* visited = (char*)memarea;
    memarea = (char*)memarea + n;

    const char* maskrow = mask;
    unsigned int start = 0;
    unsigned int maxdegree = 0;
    for (unsigned int i = 0; i < n; ++i, maskrow += nskip)
    {
        degree[i] = 0;
        visited[i] = false;
        neighbors[i] = (unsigned int*)memarea;
        for (unsigned int j = 0; j < n; ++j)
        {
            if (i != j && maskrow[j])
                neighbors[i][degree[i]++] = j;
        }
        memarea = neighbors[i] + degree[i];
        if (maxdegree <= degree[i])
        {
            maxdegree = degree[i];
            start = i;
        }
    }


    // main loop
    for (unsigned int i = 0; i < n; ++i) p[i] = (unsigned int)-1;
    int pid = n - 1;
    int nextid = pid;
    int nextstart = n - 1;
    p[pid--] = start;
    visited[start] = 1;

    unsigned int* order = (unsigned int*)memarea;
    memarea = (unsigned int*)memarea + n;

    //// sort neighbors by degree
    //SortByWeight(degreeorder, degreeorder + n, degree);
    //for (unsigned int i = 0; i < n; ++i)
    //{
    //    unsigned int *nid = neighbors[i];        
    //    SortByWeight(nid, nid + degree[i], degree);
    //}

    while (pid >= 0 && nextid >= 0)
    {
        unsigned int xi = p[nextid--];
        if (xi == (unsigned int)-1)   // we should restart at another connected conponent
        {
            maxdegree = 0;
            start = 0;
            for (unsigned int i = 0; i < n; ++i)
            {
                if (visited[i] == 0)
                {
                    if (maxdegree <= degree[i])
                    {
                        start = i;
                        maxdegree = degree[i];
                    }
                }
            }

            p[pid--] = start;
            visited[start] = 1;
            xi = start;
        }

        // sort xi's neighbors by degree, use linear insertion
        unsigned int* xneighbors = neighbors[xi];
        unsigned int d = degree[xi];
        unsigned int cnt = 0;
        for (unsigned int i = 0; i < d; ++i)
        {
            if (visited[xneighbors[i]] == 0)
                order[cnt++] = xneighbors[i];
        }

        if (cnt)
        {
            SortByWeight(order, order + cnt, degree);
            for (unsigned int i = 0; i < cnt; ++i)
            {
                p[pid--] = order[i];
                visited[order[i]] = 1;
            }
        }
    }

    return 0;
}


// solve Lh = c, where L is a lower triangle matrix
//     1 2 3  4 = heading
//            0    1    2
// L = * * * [1/a11        ]
//     * * * [a21 1/a22    ]
//     * * * [a31 a32 1/a33]
void solveLhc(
    In unsigned int n,
    In unsigned int heading,
    In const dReal* L,
    In const dReal* c,
    Out dReal* h
)
{
    h[0] = c[0] * L[0];

    L += heading;

    for (unsigned int i = 1; i < n; ++i)
    {
        dReal s = c[i];

        for (unsigned int j = 0; j < i; ++j)
            s -= L[j] * h[j];

        h[i] = s * L[i];
        L += i + heading;
    }
}

// inversion of a sparse SPD matrix, using Cholesky factorization
int sparseInverse(
    In const dReal* mat, // a sparse SPD matrix
    In const char* matMask, // the outline mask, 0-entry is zero
    In unsigned int n, // dim
    In unsigned int nskip, // rows and row skip
    InOpt unsigned int* p, // permutation matrix for better performance
    Out dReal* invmat, // nxn square matrix
    In unsigned int ninvskip,
    In void* memarea   // pre-allocated memory space for internal usage, if this parameter is zero, the function will estimate a maximum usage
)
{
    if (!memarea)
    {
        int lsize = (n * (n + 1)) >> 1;
        int req = (n * 4 + 2) * sizeof(unsigned int) +
            sizeof(dReal) * (lsize + n * ninvskip) +
            sizeof(dReal*) * (n * 2);
        return req;
    }

    /////////////////////////////////////////////
    // permutation for acceleration, nrow entries. we will perform Cholesky factorization for PAP' instead of A    
    // find the permutation
    if (!p)
    {
        p = (unsigned int*)memarea;
        memarea = (unsigned int*)memarea + n;
        for (unsigned int i = 0; i < n; ++i)
            p[i] = i;
    }

    //////////////////////////////////////////// 
    unsigned int* invp = (unsigned int*)memarea;
    memarea = (unsigned int*)memarea + n;
    for (unsigned int i = 0; i < n; ++i)
        invp[p[i]] = i;


    unsigned int* diagids = (unsigned int*)memarea;
    memarea = (unsigned int*)memarea + (n + 2);
    for (unsigned int i = 0; i < n + 2; ++i)
        diagids[i] = (i * (i + 1)) >> 1;


    // Cholesky factorization
    const dReal* c = NULL;
    dReal* h = NULL;

    unsigned int lsize = diagids[n];
    dReal* L = (dReal*)memarea;
    memarea = (dReal*)memarea + lsize;
    memset(L, 0, sizeof(dReal) * lsize);

    unsigned int* nzs = (unsigned int*)memarea;
    memarea = (unsigned int*)memarea + n;

    nzs[0] = 0;
    L[0] = dReal(1.0) / sqrt(mat[p[0] * nskip + p[0]]);

    for (unsigned int i = 1; i < n; ++i)
    {
        unsigned int base = p[i] * nskip;

        unsigned int lbase = diagids[i];
        unsigned int nz = -1;
        for (unsigned int j = 0; j < i; ++j)
        {
            if (matMask[base + p[j]])
            {
                if (nz > n)
                    nz = j;

                L[lbase + j] = mat[base + p[j]];
            }
        }
        if (nz > n)
            nz = 0;

        nzs[i] = nz;

        h = L + (lbase + nz);
        c = h;

        unsigned lstart = diagids[nz + 1] - 1;
        solveLhc(i - nz, nz + 1, L + lstart, c, h);

        // L(i, i) = (aii - h'h)^1/2
        dReal aii = mat[base + p[i]];
        for (unsigned int j = 0; j < i - nz; ++j)
            aii -= h[j] * h[j];
        L[lbase + i] = dReal(1.0) / sqrt(aii);
    }

    // solve P'LL'Px = I to compute inverse A
    // 0. initialize invMat to zero
    memset(invmat, 0, sizeof(dReal) * n * ninvskip);

    dReal** Lrows = (dReal**)memarea;
    memarea = (dReal**)memarea + n;
    for (unsigned int i = 0; i < n; ++i)
        Lrows[i] = L + diagids[i];

    // 1. solve P'Ly = I, i.e. Ly = PI
    dReal* Yt = (dReal*)memarea;
    memarea = (dReal*)memarea + n * ninvskip;
    memset(Yt, 0, sizeof(dReal) * n * ninvskip);

    dReal* Ycol = Yt;
    for (unsigned int i = 0; i < n; ++i, Ycol += ninvskip)
    {
        // 0 <= j < invp[i] is zero, so the corresponding Y are all zero
        unsigned int onepos = invp[i];
        Ycol[onepos] = 1.0 * Lrows[onepos][onepos];

        for (unsigned int j = onepos + 1; j < n; ++j)
        {
            double res = 0;
            unsigned int k = onepos > nzs[j] ? onepos : nzs[j];
            dReal* Lrowj = Lrows[j];
            for (; k < j; ++k)
                res -= Lrowj[k] * Ycol[k];

            if (res != 0)
                Ycol[j] = res * Lrowj[j];
        }
    }

    // 2. solve L'Px = y
    dReal** invMatRows = (dReal**)memarea;
    memarea = (dReal**)memarea + n;
    invMatRows[0] = invmat;
    for (unsigned int i = 1; i < n; ++i)
        invMatRows[i] = invMatRows[i - 1] + ninvskip;

    Ycol = Yt;
    for (unsigned int i = 0; i < n; ++i, Ycol += ninvskip)
    {
        unsigned int j = n - 1;
        dReal e = invMatRows[p[j]][i] = Ycol[j] * Lrows[j][j];
        invMatRows[i][p[j]] = e;
        while (j > 0)
        {
            --j;
            if (invMatRows[p[j]][i] != 0)
                continue;

            dReal res = Ycol[j];
            for (unsigned int k = n - 1; k > j; --k)
            {
                res -= invMatRows[p[k]][i] * Lrows[k][j];
            }

            e = invMatRows[p[j]][i] = (res * Lrows[j][j]);
            invMatRows[i][p[j]] = e;
        }
    }

    return 0;
}

// inversion of a sparse SPD matrix, using Cholesky factorization
int sparseInverse(
    In const dReal* mat, // a sparse SPD matrix
    In const char* matMask, // the outline mask, 0-entry is zero
    In unsigned int n, // dim
    In unsigned int nskip, // rows and row skip
    Out dReal* invmat, // nxn square matrix
    In unsigned int ninvskip,
    In void* memarea   // pre-allocated memory space for internal usage, if this parameter is zero, the function will estimate a maximum usage
)
{
    if (!memarea)
    {
        int lsize = (n * (n + 1)) >> 1;
        int req = (n * 2 + 2) * sizeof(unsigned int) +
            sizeof(dReal) * (lsize + n * ninvskip) +
            sizeof(dReal*) * (n * 2);
        return req;
    }

    unsigned int* diagids = (unsigned int*)memarea;
    memarea = (unsigned int*)memarea + (n + 2);
    for (unsigned int i = 0; i < n + 2; ++i)
        diagids[i] = (i * (i + 1)) >> 1;


    // Cholesky factorization
    const dReal* c = NULL;
    dReal* h = NULL;

    unsigned int lsize = diagids[n];
    dReal* L = (dReal*)memarea;
    memarea = (dReal*)memarea + lsize;
    memset(L, 0, sizeof(dReal) * lsize);

    unsigned int* nzs = (unsigned int*)memarea;
    memarea = (unsigned int*)memarea + n;

    nzs[0] = 0;
    L[0] = dReal(1.0) / sqrt(mat[0]);

    for (unsigned int i = 1; i < n; ++i)
    {
        unsigned int base = i * nskip;

        unsigned int lbase = diagids[i];
        unsigned int nz = -1;
        for (unsigned int j = 0; j < i; ++j)
        {
            if (matMask[base + j])
            {
                if (nz > n)
                    nz = j;

                L[lbase + j] = mat[base + j];
            }
        }
        if (nz > n)
            nz = 0;

        nzs[i] = nz;

        h = L + (lbase + nz);
        c = h;

        unsigned lstart = diagids[nz + 1] - 1;
        solveLhc(i - nz, nz + 1, L + lstart, c, h);

        // L(i, i) = (aii - h'h)^1/2
        dReal aii = mat[base + i];
        for (unsigned int j = 0; j < i - nz; ++j)
            aii -= h[j] * h[j];

        // we store inverse of the diag elements
        L[lbase + i] = dReal(1.0) / sqrt(aii);
    }

#if 0
    {
        std::ofstream fout("L.txt");
        fout.precision(18);
        for (unsigned int i = 0; i < n; ++i)
        {
            dReal* Lrow = L + diagids[i];
            for (unsigned int j = 0; j < i; ++j)
                fout << Lrow[j] << " ";
            fout << 1.0 / Lrow[i] << " ";
            for (unsigned int j = i + 1; j < n; ++j)
                fout << 0.0 << " ";
            fout << std::endl;
        }
    }
#endif

    // solve P'LL'Px = I to compute inverse A
    // 0. initialize invMat to zero

    dReal** Lrows = (dReal**)memarea;
    memarea = (dReal**)memarea + n;
    for (unsigned int i = 0; i < n; ++i)
        Lrows[i] = L + diagids[i];

    // 1. solve P'Ly = I, i.e. Ly = PI
    dReal* Yt = (dReal*)memarea;
    memarea = (dReal*)memarea + n * ninvskip;
    //memset(Yt, 0, sizeof(dReal) * n * ninvskip);

    dReal* Ycol = Yt;
    for (unsigned int i = 0; i < n; ++i, Ycol += ninvskip)
    {
        // 0 <= j < i is zero, so the corresponding Y are all zero
        for (unsigned int j = 0; j < i; ++j)
            Ycol[j] = 0.0;

        Ycol[i] = 1.0 * Lrows[i][i];

        for (unsigned int j = i + 1; j < n; ++j)
        {
            double res = 0;
            unsigned int k = i > nzs[j] ? i : nzs[j];
            dReal* Lrowj = Lrows[j];
            for (; k < j; ++k)
                res -= Lrowj[k] * Ycol[k];

            if (res != 0)
                Ycol[j] = res * Lrowj[j];
            else
                Ycol[j] = 0.0;
        }
    }

    // 2. solve L'Px = y
    //memset(invmat, 0, sizeof(dReal) * n * ninvskip);
    dReal** invMatRows = (dReal**)memarea;
    memarea = (dReal**)memarea + n;
    invMatRows[0] = invmat;
    for (unsigned int i = 1; i < n; ++i)
        invMatRows[i] = invMatRows[i - 1] + ninvskip;

    Ycol = Yt;
    for (unsigned int i = 0; i < n; ++i, Ycol += ninvskip)
    {
        unsigned int j = n - 1;
        dReal e = invMatRows[j][i] = Ycol[j] * Lrows[j][j];
        invMatRows[i][j] = e;
        while (j > i)
        {
            --j;

            dReal res = Ycol[j];
            for (unsigned int k = n - 1; k > j; --k)
                res -= invMatRows[k][i] * Lrows[k][j];

            e = invMatRows[j][i] = (res * Lrows[j][j]);
            invMatRows[i][j] = e;
        }
    }

    return 0;
}