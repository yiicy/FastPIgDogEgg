#ifndef _COMMOM_H_
#define _COMMOM_H_

#include <vector>
#include <cmath>
#include <string>
#include <map>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <arm_neon.h>

#ifdef USE_OPENMP
#include <omp.h>
#endif // omp
namespace PigDogEgg{

#   define LOGI(fmt, ...) printf(fmt, ##__VA_ARGS__)
#   if defined(ENABLE_DEBUG)
#       define LOGD(fmt, ...) printf(fmt, ##__VA_ARGS__)
#       define LOGW(fmt, ...) printf(fmt, ##__VA_ARGS__)
#   else
#       define LOGD(fmt, ...)
#       define LOGW(fmt, ...)
#   endif
#   define LOGE(fmt, ...) printf(fmt, ##__VA_ARGS__)
#   define LOGF(fmt, ...) printf(fmt, ##__VA_ARGS__); assert(0)

#define CHECK_EQ(a, b, fmt, ...) \
do { if ((a) != (b)) { LOGF(fmt, ##__VA_ARGS__);} } while (0)

#define CHECK_GE(a, b, fmt, ...) \
do { if ((a) < (b)) { LOGF(fmt, ##__VA_ARGS__);} } while (0)

#define CHECK_GT(a, b, fmt, ...) \
do { if ((a) <= (b)) { LOGF(fmt, ##__VA_ARGS__);} } while (0)

#define CHECK_LE(a, b, fmt, ...) \
do { if ((a) > (b)) { LOGF(fmt, ##__VA_ARGS__);} } while (0)

#define CHECK_LT(a, b, fmt, ...) \
do { if ((a) >= (b)) { LOGF(fmt, ##__VA_ARGS__);} } while (0)

typedef enum{
    FpdeOk = 0,
    FpdeFail = 1
} FpdeStatus;
//! the alignment of all the allocated buffers
const int MALLOC_ALIGN = 16;

static double FpdeTimer()
{
    struct timeval tv;    
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000.0) + (tv.tv_usec / 1000.0);
}

static inline void* fast_malloc(size_t size) {
    size_t offset = sizeof(void*) + MALLOC_ALIGN - 1;
    char* p;
    p = static_cast<char*>(malloc(offset + size));
    if (!p) {
        return nullptr;
    }
    void* r = reinterpret_cast<void*>(reinterpret_cast<size_t>(p + offset) & (~(MALLOC_ALIGN - 1)));
    static_cast<void**>(r)[-1] = p;
    return r;
}

static inline void fast_free(void* ptr) {
    if (ptr){
        free(static_cast<void**>(ptr)[-1]);
    }
}

}
#endif // _COMMON_H_