#ifndef _FPDE_H_
#define _FPDE_H_
#include "common.h"

namespace PigDogEgg{
    extern FpdeStatus FpdeMiniMatMulQuantize4x4(const float* A, const float* B, float* C, \
        									    const int M, const int N, const int K, const float scale);

	extern FpdeStatus FpdeMiniMatMul4x4(const float* A, const float* B, float* C, \
                                        const int M, const int N, const int K);

	extern FpdeStatus FpdeMiniMatMul4x3(const float* A, const float* B, float* C, \
                                        const int M, const int N, const int K);
}

#endif // _FPDE_H_