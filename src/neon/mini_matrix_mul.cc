#include "fpde.h"

namespace PigDogEgg{

//! mat A~[0.f, 1.f], mat B~[0.f, 1.f], scale~[0.f, 127.f]
FpdeStatus FpdeMiniMatMulQuantize4x4(const float* A, const float* B, float* C, \
                                  const int M, const int N, const int K, const float scale){
    if (N != 4 || K != 4){
        LOGE("N and K must be 4\n");
        return FpdeFail;
    }

    int cnt = (M >> 3);
    int rem = (M & 7);
    float* c_ptr = C;
    const float* a_ptr = A;
    const float* b_ptr = B;
    float32x4_t vscale = vdupq_n_f32(scale);
    float32x4_t viscale = vdupq_n_f32(1.f / (scale * scale));

#ifdef __aarch64__
    asm volatile(
        //! load b
        "ld1 {v0.4s, v1.4s}, [%[b]], #32  \n"
        "ld1 {v2.4s, v3.4s}, [%[b]]  \n"

        //！ quantize b
        "fmul v0.4s, v0.4s, %[vscale].4s  \n"
        "fmul v1.4s, v1.4s, %[vscale].4s  \n"
        "fmul v2.4s, v2.4s, %[vscale].4s  \n"
        "fmul v3.4s, v3.4s, %[vscale].4s  \n"

        "ld4 {v6.4s,  v7.4s,  v8.4s,  v9.4s},  [%[a]], #64  \n"
        "ld4 {v10.4s, v11.4s, v12.4s, v13.4s}, [%[a]], #64  \n"

        "fcvtau v0.4s, v0.4s  \n"
        "fcvtau v1.4s, v1.4s  \n"
        "fcvtau v2.4s, v2.4s  \n"
        "fcvtau v3.4s, v3.4s  \n"

        //! convert b to uint16
        "uqxtn  v4.4h, v0.4s  \n"
        "uqxtn2 v4.8h, v1.4s  \n"
        "uqxtn  v5.4h, v2.4s  \n"
        "uqxtn2 v5.8h, v3.4s  \n"
        
        "cbz   %w[cnt], 2f\n"
        "1:  \n"

        //! convert a to uint16
        "fmul v6.4s,  v6.4s,  %[vscale].4s  \n"
        "fmul v7.4s,  v7.4s,  %[vscale].4s  \n"
        "fmul v8.4s,  v8.4s,  %[vscale].4s  \n"
        "fmul v9.4s,  v9.4s,  %[vscale].4s  \n"
        "fmul v10.4s, v10.4s, %[vscale].4s  \n"
        "fmul v11.4s, v11.4s, %[vscale].4s  \n"
        "fmul v12.4s, v12.4s, %[vscale].4s  \n"
        "fmul v13.4s, v13.4s, %[vscale].4s  \n"

        "fcvtau v6.4s,  v6.4s   \n"
        "fcvtau v7.4s,  v7.4s   \n"
        "fcvtau v8.4s,  v8.4s   \n"
        "fcvtau v9.4s,  v9.4s   \n"
        "fcvtau v10.4s, v10.4s  \n"
        "fcvtau v11.4s, v11.4s  \n"
        "fcvtau v12.4s, v12.4s  \n"
        "fcvtau v13.4s, v13.4s  \n"

        "uqxtn  v0.4h, v6.4s   \n"
        "uqxtn2 v0.8h, v10.4s  \n"
        "uqxtn  v1.4h, v7.4s   \n"
        "uqxtn2 v1.8h, v11.4s  \n"
        "uqxtn  v2.4h, v8.4s   \n"
        "uqxtn2 v2.8h, v12.4s  \n"
        "uqxtn  v3.4h, v9.4s   \n"
        "uqxtn2 v3.8h, v13.4s  \n"

        // deal with mat(8x4) * mat(4x4) 
        "mul v14.8h, v0.8h, v4.h[0]  \n"
        "mul v15.8h, v0.8h, v4.h[1]  \n"
        "mul v16.8h, v0.8h, v4.h[2]  \n"
        "mul v17.8h, v0.8h, v4.h[3]  \n"

        "mla v14.8h, v1.8h, v4.h[4]  \n"
        "mla v15.8h, v1.8h, v4.h[5]  \n"
        "mla v16.8h, v1.8h, v4.h[6]  \n"
        "mla v17.8h, v1.8h, v4.h[7]  \n"

        "mla v14.8h, v2.8h, v5.h[0]  \n"
        "mla v15.8h, v2.8h, v5.h[1]  \n"
        "mla v16.8h, v2.8h, v5.h[2]  \n"
        "mla v17.8h, v2.8h, v5.h[3]  \n"

        "mla v14.8h, v3.8h, v5.h[4]  \n"
        "mla v15.8h, v3.8h, v5.h[5]  \n"
        "mla v16.8h, v3.8h, v5.h[6]  \n"
        "mla v17.8h, v3.8h, v5.h[7]  \n"

        //! convet to uint32
        "ushll  v6.4s,  v14.4h, #0  \n"
        "ushll  v7.4s,  v15.4h, #0  \n"
        "ushll  v8.4s,  v16.4h, #0  \n"
        "ushll  v9.4s,  v17.4h, #0  \n"

        "ushll2 v10.4s, v14.8h, #0  \n"
        "ushll2 v11.4s, v15.8h, #0  \n"
        "ushll2 v12.4s, v16.8h, #0  \n"
        "ushll2 v13.4s, v17.8h, #0  \n"

        //! convert to float
        "ucvtf v14.4s, v6.4s   \n"
        "ucvtf v15.4s, v7.4s   \n"
        "ucvtf v16.4s, v8.4s   \n"
        "ucvtf v17.4s, v9.4s   \n"

        "ucvtf v18.4s, v10.4s  \n"
        "ucvtf v19.4s, v11.4s  \n"
        "ucvtf v20.4s, v12.4s  \n"
        "ucvtf v21.4s, v13.4s  \n"

        "ld4 {v6.4s,  v7.4s,  v8.4s,  v9.4s},  [%[a]], #64  \n"
        "ld4 {v10.4s, v11.4s, v12.4s, v13.4s}, [%[a]], #64  \n"

        //！ dequantize
        "fmul v14.4s, v14.4s, %[viscale].4s  \n"
        "fmul v15.4s, v15.4s, %[viscale].4s  \n"
        "fmul v16.4s, v16.4s, %[viscale].4s  \n"
        "fmul v17.4s, v17.4s, %[viscale].4s  \n"

        "subs %w[cnt], %w[cnt], #1  \n"

        "fmul v18.4s, v18.4s, %[viscale].4s  \n"
        "fmul v19.4s, v19.4s, %[viscale].4s  \n"
        "fmul v20.4s, v20.4s, %[viscale].4s  \n"
        "fmul v21.4s, v21.4s, %[viscale].4s  \n"

        //! store c
        "st4 {v14.4s, v15.4s, v16.4s, v17.4s}, [%[c]], #64  \n"
        "st4 {v18.4s, v19.4s, v20.4s, v21.4s}, [%[c]], #64  \n"
        "bne 1b  \n"

        "2:  \n"
        "cbz  %w[rem], 4f           \n"
        "sub  %[a],    %[a], #128   \n"
        "ld1  {v0.4s}, [%[a]], #16  \n"
        "dup  v6.2d,   v4.d[1]      \n"
        "dup  v7.2d,   v5.d[1]      \n"

        "3:  \n"
        //！ quantize a
        "fmul v0.4s, v0.4s, %[vscale].4s  \n"
        //! convert a to uint16
        "fcvtau v1.4s, v0.4s  \n"
        "uqxtn  v2.4h, v1.4s  \n"

        //! deal with mat(1x4) * mat(4x4)
        "mul  v8.4h, v4.4h, v2.h[0]  \n"
        "mul  v9.4h, v5.4h, v2.h[2]  \n"
        "mla  v8.4h, v6.4h, v2.h[1]  \n"
        "mla  v9.4h, v7.4h, v2.h[3]  \n"
        "add  v10.4h, v8.4h, v9.4h   \n"

        "subs  %w[rem], %w[rem], #1   \n"
        "ld1   {v0.4s}, [%[a]],  #16  \n"
        //! convet to uint32
        "ushll v11.4s,  v10.4h,  #0   \n"

        //! convert to float
        "ucvtf v11.4s, v11.4s         \n"

        //！ dequantize
        "fmul v11.4s,   v11.4s, %[viscale].4s  \n"
        "st1  {v11.4s}, [%[c]], #16  \n"
        "bne  3b                     \n"

        "4:                          \n"
        : [c] "+r" (c_ptr), [a] "+r" (a_ptr), [b] "+r" (b_ptr), [cnt] "+r" (cnt), \
          [rem] "+r" (rem)
        : [vscale] "w" (vscale), [viscale] "w" (viscale)
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", \
          "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21"
    );
    return FpdeOk;
#else // __aarch64__
    LOGE("FpdeMiniMatMulQuantize armv7 not impl\n");
    return FpdeFail;
#endif // __aarch64__
}

FpdeStatus FpdeMiniMatMul4x4(const float* A, const float* B, float* C, \
                          const int M, const int N, const int K){
    if (N != 4 || K != 4){
        LOGE("N and K must be 4\n");
        return FpdeFail;
    }

    int cnt = (M >> 2);
    int rem = (M & 3);
    float* c_ptr = C;
    const float* a_ptr = A;
    const float* b_ptr = B;
#ifdef __aarch64__
    asm volatile (
        "ld1 {v0.4s, v1.4s}, [%[b]], #32    \n"
        "ld1 {v2.4s, v3.4s}, [%[b]]         \n"
        "ld1 {v4.4s, v5.4s}, [%[a]], #32    \n"
        "ld1 {v6.4s, v7.4s}, [%[a]], #32    \n"
        "cbz %w[cnt], 2f                    \n"

        "1:                                 \n"
        "fmul v8.4s,  v0.4s, v4.s[0]        \n"
        "fmul v9.4s,  v0.4s, v5.s[0]        \n"
        "fmul v10.4s, v0.4s, v6.s[0]        \n"
        "fmul v11.4s, v0.4s, v7.s[0]        \n"

        "fmla v8.4s,  v1.4s, v4.s[1]        \n"
        "fmla v9.4s,  v1.4s, v5.s[1]        \n"
        "fmla v10.4s, v1.4s, v6.s[1]        \n"
        "fmla v11.4s, v1.4s, v7.s[1]        \n"

        "fmla v8.4s,  v2.4s, v4.s[2]        \n"
        "fmla v9.4s,  v2.4s, v5.s[2]        \n"
        "fmla v10.4s, v2.4s, v6.s[2]        \n"
        "fmla v11.4s, v2.4s, v7.s[2]        \n"

        "fmla v8.4s,  v3.4s, v4.s[3]        \n"
        "fmla v9.4s,  v3.4s, v5.s[3]        \n"
        "fmla v10.4s, v3.4s, v6.s[3]        \n"
        "fmla v11.4s, v3.4s, v7.s[3]        \n"

        "subs %w[cnt], %w[cnt], #1          \n"

        "ld1 {v4.4s, v5.4s},   [%[a]], #32  \n"
        "ld1 {v6.4s, v7.4s},   [%[a]], #32  \n"
        "st1 {v8.4s, v9.4s},   [%[c]], #32  \n"
        "st1 {v10.4s, v11.4s}, [%[c]], #32  \n"
        "bne 1b                             \n"

        "2:                                 \n"
        "cbz %w[rem], 4f                    \n"
        "sub %[a], %[a], #64                \n"
        "ld1 {v4.4s}, [%[a]], #16           \n"

        "3:                                 \n"
        "fmul v8.4s, v0.4s, v4.s[0]         \n"
        "fmul v9.4s, v1.4s, v4.s[1]         \n"
        "subs %w[rem], %w[rem], #1          \n"
        "fmla v8.4s, v2.4s, v4.s[2]         \n"
        "fmla v9.4s, v3.4s, v4.s[3]         \n"
        "ld1 {v4.4s}, [%[a]], #16           \n"
        "fadd v10.4s, v8.4s, v9.4s          \n"
        "st1 {v10.4s}, [%[c]], #16          \n"
        "bne 3b                             \n"

        "4:                                 \n"
        : [c] "+r" (c_ptr), [a] "+r" (a_ptr), [b] "+r" (b_ptr), [cnt] "+r" (cnt), \
          [rem] "+r" (rem)
        :
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",\
          "v10", "v11"
    );
#else //__aarch64__
    asm volatile (
        "vld1.32 {d0-d3},   [%[b]]!  \n"
        "vld1.32 {d4-d7},   [%[b]]   \n"
        "vld1.32 {d8-d11},  [%[a]]!  \n"
        "vld1.32 {d12-d15}, [%[a]]!  \n"

        "cmp %[cnt], #0              \n"
        "beq 2f                      \n"

        "1:                          \n"
        "vmul.f32 q8,  q0, d8[0]     \n"
        "vmul.f32 q9,  q0, d10[0]    \n"
        "vmul.f32 q10, q0, d12[0]    \n"
        "vmul.f32 q11, q0, d14[0]    \n"

        "vmla.f32 q8,  q1, d8[1]     \n"
        "vmla.f32 q9,  q1, d10[1]    \n"
        "vmla.f32 q10, q1, d12[1]    \n"
        "vmla.f32 q11, q1, d14[1]    \n"

        "vmla.f32 q8,  q2, d9[0]     \n"
        "vmla.f32 q9,  q2, d11[0]    \n"
        "vmla.f32 q10, q2, d13[0]    \n"
        "vmla.f32 q11, q2, d15[0]    \n"

        "vmla.f32 q8,  q3, d9[1]     \n"
        "vmla.f32 q9,  q3, d11[1]    \n"
        "vmla.f32 q10, q3, d13[1]    \n"
        "vmla.f32 q11, q3, d15[1]    \n"

        "subs %[cnt], #1             \n"

        "vld1.32 {d8-d11},  [%[a]]!  \n"
        "vld1.32 {d12-d15}, [%[a]]!  \n"
        "vst1.32 {d16-d19}, [%[c]]!  \n"
        "vst1.32 {d20-d23}, [%[c]]!  \n"
        "bne 1b                      \n"

        "2:                          \n"
        "cmp %[rem], #0              \n"
        "beq 4f                      \n"
        "sub %[a], #64               \n"
        "vld1.32 {d8-d9}, [%[a]]!    \n"

        "3:                          \n"
        "vmul.f32 q8, q0, d8[0]      \n"
        "vmul.f32 q9, q1, d8[1]      \n"
        "subs %[rem], #1             \n"
        "vmla.f32 q8, q2, d9[0]      \n"
        "vmla.f32 q9, q3, d9[1]      \n"
        "vld1.32 {d8-d9}, [%[a]]!    \n"
        "vadd.f32 q10, q8, q9        \n"
        "vst1.32 {d20-d21}, [%[c]]!  \n"
        "bne 3b                      \n"

        "4:                          \n"
        : [c] "+r" (c_ptr), [a] "+r" (a_ptr), [b] "+r" (b_ptr), [cnt] "+r" (cnt), \
          [rem] "+r" (rem)
        :
        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9",\
          "q10", "q11"
    );
#endif // __aarch64__
    return FpdeOk;
}

FpdeStatus FpdeMiniMatMul4x3(const float* A, const float* B, float* C, \
                          const int M, const int N, const int K){
    if (N != 3 || K != 4){
        LOGE("N must be 3 and K must be 4\n");
        return FpdeFail;
    }
    
    int cnt = (M >> 2);
    int rem = (M & 3);
    float* c_ptr = C;
    const float* a_ptr = A;
    const float* b_ptr = B;
#ifdef __aarch64__
    asm volatile (
        "ld3 {v0.4s, v1.4s, v2.4s}, [%[b]]  \n"
        "ld4 {v3.4s, v4.4s, v5.4s, v6.4s}, [%[a]], #64 \n"
        "cbz %w[cnt], 2f                    \n"

        "1:                                 \n"
        "fmul v7.4s, v3.4s, v0.s[0]         \n"
        "fmul v8.4s, v3.4s, v1.s[0]         \n"
        "fmul v9.4s, v3.4s, v2.s[0]         \n"

        "fmla v7.4s, v4.4s, v0.s[1]         \n"
        "fmla v8.4s, v4.4s, v1.s[1]         \n"
        "fmla v9.4s, v4.4s, v2.s[1]         \n"

        "fmla v7.4s, v5.4s, v0.s[2]         \n"
        "fmla v8.4s, v5.4s, v1.s[2]         \n"
        "fmla v9.4s, v5.4s, v2.s[2]         \n"

        "fmla v7.4s, v6.4s, v0.s[3]         \n"
        "fmla v8.4s, v6.4s, v1.s[3]         \n"
        "fmla v9.4s, v6.4s, v2.s[3]         \n"

        "subs %w[cnt], %w[cnt], #1          \n"

        "ld4 {v3.4s, v4.4s, v5.4s, v6.4s}, [%[a]], #64  \n"
        "st3 {v7.4s, v8.4s, v9.4s},   [%[c]], #48       \n"
        "bne 1b                                         \n"

        "2:                                 \n"
        "cbz %w[rem], 4f                    \n"
        "sub %[a], %[a], #64                \n"
        "ld1 {v3.4s}, [%[a]], #16           \n"

        "3:                                 \n"
        "fmul   v4.4s, v3.4s, v0.4s         \n"
        "fmul   v5.4s, v3.4s, v1.4s         \n"
        "fmul   v6.4s, v3.4s, v2.4s         \n"
        "faddp  v7.4s, v4.4s, v5.4s         \n"
        "faddp  v8.4s, v6.4s, v6.4s         \n"
        "ld1    {v3.4s}, [%[a]], #16        \n"
        "subs   %w[rem], %w[rem], #1        \n"
        "faddp  v9.4s, v7.4s, v8.4s         \n"
        "dup    v10.2d, v9.d[1]             \n"
        "str    d9, [%[c]], #8              \n"
        "str    s10, [%[c]], #4             \n"
        "bne    3b                          \n"

        "4:                                 \n"
        : [c] "+r" (c_ptr), [a] "+r" (a_ptr), [b] "+r" (b_ptr), [cnt] "+r" (cnt), \
          [rem] "+r" (rem)
        :
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"
    );
#else //__aarch64__

    asm volatile (
        "vld3.32 {d0, d2, d4}, [%[b]]!  \n"
        "vld3.32 {d1, d3, d5}, [%[b]]   \n"
        "vld4.32 {d6, d8, d10, d12}, [%[a]]!  \n"
        "vld4.32 {d7, d9, d11, d13}, [%[a]]!  \n"
        "cmp %[cnt], #0                 \n"
        "beq 2f                         \n"

        "1:                             \n"
        "vmul.f32 q7, q3, d0[0]         \n"
        "vmul.f32 q8, q3, d2[0]         \n"
        "vmul.f32 q9, q3, d4[0]         \n"

        "vmla.f32 q7, q4, d0[1]         \n"
        "vmla.f32 q8, q4, d2[1]         \n"
        "vmla.f32 q9, q4, d4[1]         \n"

        "vmla.f32 q7, q5, d1[0]         \n"
        "vmla.f32 q8, q5, d3[0]         \n"
        "vmla.f32 q9, q5, d5[0]         \n"

        "vmla.f32 q7, q6, d1[1]         \n"
        "vmla.f32 q8, q6, d3[1]         \n"
        "vmla.f32 q9, q6, d5[1]         \n"

        "subs %[cnt], #1                \n"

        "vld4.32 {d6, d8, d10, d12}, [%[a]]!  \n"
        "vld4.32 {d7, d9, d11, d13}, [%[a]]!  \n"
        "vst3.32 {d14, d16, d18}, [%[c]]!     \n"
        "vst3.32 {d15, d17, d19}, [%[c]]!     \n"
        "bne 1b                               \n"

        "2:                             \n"
        "cmp %[rem], #0                 \n"
        "beq 4f                         \n"
        "sub %[a], #64                  \n"
        "vld1.32 {d6-d7}, [%[a]]!       \n"

        "3:                             \n"
        "vmul.f32 q4, q3, q0            \n"
        "vmul.f32 q5, q3, q1            \n"
        "subs %[rem], #1                \n"
        "vmul.f32 q6, q3, q2            \n"
        "vpadd.f32 d14, d8, d9          \n"
        "vld1.32 {d6-d7}, [%[a]]!       \n"
        "vpadd.f32 d15, d10, d11        \n"
        "vpadd.f32 d16, d12, d13        \n"
        "vpadd.f32 d18, d14, d15        \n"
        "vpadd.f32 d19, d16, d16        \n"
        "vst1.32 {d18}, [%[c]]!         \n"
        "vst1.32 {d19[0]}, [%[c]]!      \n"
        "bne 3b                         \n"

        "4:                             \n"
        : [c] "+r" (c_ptr), [a] "+r" (a_ptr), [b] "+r" (b_ptr), [cnt] "+r" (cnt), \
          [rem] "+r" (rem)
        : 
        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9"
    );

#endif // __aarch64__
    return FpdeOk;
}


}