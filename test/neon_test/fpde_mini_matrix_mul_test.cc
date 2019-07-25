#include "fpde.h"
using namespace PigDogEgg;

void BasicMiniMatMul(const float *A, const float *B, float *C, int M, int K, int N)
{
    for (int m=0; m<M; m++) 
    {
        for (int n=0; n<N; n++) 
        {
            float acc = 0.0f;
            for (int k=0; k<K; k++) 
            {
                acc += A[m*K + k] * B[k*N + n];
            }
            C[m * N + n] = acc;
        }
    }
}

static void fill_mat_const(float* mat, int size, float val){
    for( int i = 0; i < size; ++i){
        mat[i] = val;
    }
}

static void fill_mat_rand(float* mat, int size, float min, float max){
    srand(time(NULL));
    for (int i = 0; i < size; ++i){
        mat[i] = (float)rand() / RAND_MAX * (max - min) + min;
    }
}

static void print_mat(const float* mat, int m, int n){
    for (int i = 0; i < m; ++i){
        for (int j = 0; j < n; ++j){
            printf("%f ", mat[i * n + j]);
        }
        printf("\n");
    }
}

static void mat_cmp(const float* mat1, const float* mat2, const int size, \
                        float* max_diff, float* max_ratio){
    *max_diff = 0.f;
    *max_ratio = 0.f;
    for (int i = 0; i < size; ++i){
        float diff = std::abs(mat1[i] - mat2[i]);
        float ratio = diff / std::abs(mat1[i]);
        if (diff > *max_diff){
            *max_diff = diff;
        }
        if (ratio > *max_ratio){
            *max_ratio = ratio;
        }
    }
}

static void mat_diff(const float* mat1, const float* mat2, float* diff_mat, const int size){
    if (diff_mat == nullptr){
        LOGE("diff mat can't be null\n");
        return;
    }
    for (int i = 0; i < size; ++i){
        diff_mat[i] = std::abs(mat1[i] - mat2[i]);
    }
}

int M = 20;
int N = 4;
int K = 4;
int iter = 100;
float scale = 100.f;


void test_fpde_mini_matmul4x4(){
    if (N != 4){
        LOGE("N != 4, skip fpde mini matmul 4x4 test\n");
        return;
    }
    float* a = (float*)fast_malloc(M * K * sizeof(float));
    float* b = (float*)fast_malloc(K * N * sizeof(float));
    float* c = (float*)fast_malloc(M * N * sizeof(float));
    float* basic_c = (float*)fast_malloc(M * N * sizeof(float));
    float* diff_c = (float*)fast_malloc(M * N * sizeof(float));

    //fill_mat_const(a, M * K, 0.2);
    //fill_mat_const(b, K * N, 0.2);
    fill_mat_rand(a, M * K, 0.f, 1.f);
    fill_mat_rand(b, K * N, 0.f, 1.f);

    // basic
    double basic_start = FpdeTimer();
    for (int i = 0; i < iter; ++i){
        BasicMiniMatMul(a, b, basic_c, M, K, N);
    }
    double basic_elapse = FpdeTimer() - basic_start;

    // fpde
    FpdeStatus ret;
    double fpde_start = FpdeTimer();
    for (int i = 0; i < iter; ++i){
        ret = FpdeMiniMatMul4x4(a, b, c, M, N, K);
    }
    double fpde_elapse = FpdeTimer() - fpde_start;
    if (ret != FpdeOk){
        LOGE("run fpde mini matrix mul 4x4 fail!!\n");
    }

    // LOGI("fpde result: \n");
    // print_mat(c, M, N);
    // LOGI("basic result: \n");
    // print_mat(basic_c, M, N);
    // LOGI("mini mat mul: \n");

    float max_diff, max_ratio;
    mat_cmp(c, basic_c, M * N, &max_diff, &max_ratio);
    if (max_ratio > 0.01) {
        if (max_diff > 0.1) {
            mat_diff(c, basic_c, diff_c, M * N);
            LOGE("ERROR: 4x4 NO PASS!!!, max diff: %.5f, max ratio: %.5f\n", max_diff, max_ratio);
            LOGE("diff result: \n");
            print_mat(diff_c, M, N);
        } else {
            LOGI("4x4 PASS!!, max diff: %.5f, max ratio: %.5f\n", max_diff, max_ratio);
        }
    }  else {
         LOGI("4x4 PASS!!, max diff: %.5f, max ratio: %.5f\n", max_diff, max_ratio);
    }
   
    LOGI("4x4 basic average time: %.5f ms\n", basic_elapse / iter);
    LOGI("4x4 fpde average time: %.5f ms\n",  fpde_elapse / iter );
    fast_free(a);
    fast_free(b);
    fast_free(c);
    fast_free(diff_c);
    fast_free(basic_c);
}

void test_fpde_mini_matmul_quantize4x4(){
    if (N != 4){
        LOGE("N != 4, skip fpde mini matmul quantize 4x4 test\n");
        return;
    }
#ifdef __aarch64__
    float* a = (float*)fast_malloc(M * K * sizeof(float));
    float* b = (float*)fast_malloc(K * N * sizeof(float));
    float* c = (float*)fast_malloc(M * N * sizeof(float));
    float* basic_c = (float*)fast_malloc(M * N * sizeof(float));
    float* diff_c = (float*)fast_malloc(M * N * sizeof(float));
    
    //fill_mat_const(a, M * K, 0.2);
    //fill_mat_const(b, K * N, 0.2);
    fill_mat_rand(a, M * K, 0.f, 1.f);
    fill_mat_rand(b, K * N, 0.f, 1.f);

    // basic
    double basic_start = FpdeTimer();
    for (int i = 0; i < iter; ++i){
        BasicMiniMatMul(a, b, basic_c, M, K, N);
    }
    double basic_elapse = FpdeTimer() - basic_start;

    // fpde
    FpdeStatus ret;
    double fpde_start = FpdeTimer();
    for (int i = 0; i < iter; ++i){
        ret = FpdeMiniMatMulQuantize4x4(a, b, c, M, N, K, scale);
    }
    double fpde_elapse = FpdeTimer() - fpde_start;
    if (ret != FpdeOk){
        LOGE("run fpde mini matrix quantize mul 4x4 fail!!\n");
    }

    // LOGI("fpde result: \n");
    // print_mat(c, M, N);
    // LOGI("basic result: \n");
    // print_mat(basic_c, M, N);
    // LOGI("mini mat mul: \n");

    float max_diff, max_ratio;
    mat_cmp(c, basic_c, M * N, &max_diff, &max_ratio);
    if (max_ratio > 0.01) {
        if (max_diff > 0.1) {
            mat_diff(c, basic_c, diff_c, M * N);
            LOGE("ERROR: 4x4 quantize NO PASS!!!, max diff: %.5f, max ratio: %.5f\n", max_diff, max_ratio);
            LOGE("diff result: \n");
            print_mat(diff_c, M, N);
        } else {
            LOGI("4x4 quantize PASS!!, max diff: %.5f, max ratio: %.5f\n", max_diff, max_ratio);
        }
    }  else {
         LOGI("4x4 quantize PASS!!, max diff: %.5f, max ratio: %.5f\n", max_diff, max_ratio);
    }
   
    LOGI("4x4 basic average time: %.5f ms\n", basic_elapse / iter);
    LOGI("4x4 quantize fpde average time: %.5f ms\n",  fpde_elapse / iter );

    fast_free(a);
    fast_free(b);
    fast_free(c);
    fast_free(diff_c);
    fast_free(basic_c);
#else // __aarch64__
    LOGE("test_fpde_mini_matmul_quantize_4x4 not support armv7\n");
#endif // __aarch64__
}

void test_fpde_mini_matmul4x3(){
    if (N != 3){
        LOGE("N != 3, skip fpde mini matmul 4x3 test\n");
        return;
    }
    float* a = (float*)fast_malloc(M * K * sizeof(float));
    float* b = (float*)fast_malloc(K * N * sizeof(float));
    float* c = (float*)fast_malloc(M * N * sizeof(float));
    float* basic_c = (float*)fast_malloc(M * N * sizeof(float));
    float* diff_c = (float*)fast_malloc(M * N * sizeof(float));

    //fill_mat_const(a, M * K, 0.2);
    //fill_mat_const(b, K * N, 0.2);
    fill_mat_rand(a, M * K, 0.f, 1.f);
    fill_mat_rand(b, K * N, 0.f, 1.f);

    // basic
    double basic_start = FpdeTimer();
    for (int i = 0; i < iter; ++i){
        BasicMiniMatMul(a, b, basic_c, M, K, N);
    }
    double basic_elapse = FpdeTimer() - basic_start;

    // fpde
    FpdeStatus ret;
    double fpde_start = FpdeTimer();
    for (int i = 0; i < iter; ++i){
        ret = FpdeMiniMatMul4x3(a, b, c, M, N, K);
    }
    double fpde_elapse = FpdeTimer() - fpde_start;
    if (ret != FpdeOk){
        LOGE("run fpde mini matrix mul 4x4 fail!!\n");
    }

    // LOGI("fpde result: \n");
    // print_mat(c, M, N);
    // LOGI("basic result: \n");
    // print_mat(basic_c, M, N);
    // LOGI("mini mat mul: \n");

    float max_diff, max_ratio;
    mat_cmp(c, basic_c, M * N, &max_diff, &max_ratio);
    if (max_ratio > 0.01) {
        if (max_diff > 0.1) {
            mat_diff(c, basic_c, diff_c, M * N);
            LOGE("ERROR: 4x3 NO PASS!!!, max diff: %.5f, max ratio: %.5f\n", max_diff, max_ratio);
            LOGE("diff result: \n");
            print_mat(diff_c, M, N);
        } else {
            LOGI("4x3 PASS!!, max diff: %.5f, max ratio: %.5f\n", max_diff, max_ratio);
        }
    }  else {
         LOGI("4x3 PASS!!, max diff: %.5f, max ratio: %.5f\n", max_diff, max_ratio);
    }
   
    LOGI("4x3 basic average time: %.5f ms\n", basic_elapse / iter);
    LOGI("4x3 fpde average time: %.5f ms\n",  fpde_elapse / iter );
    fast_free(a);
    fast_free(b);
    fast_free(c);
    fast_free(diff_c);
    fast_free(basic_c);
}
int main(int argc, const char** argv){
    if (argc > 1){
        M = atoi(argv[1]);
    }

    if (argc > 2){
        N = atoi(argv[2]);
    }

    if (argc > 3){
        K = atoi(argv[3]);
    }

    if (argc > 4){
        scale = atof(argv[4]);
    }
    if (argc > 5){
        iter = atoi(argv[5]);
    }
    LOGI("M = %d, N = %d, K = %d, scale = %.5f, iter = %d\n\n", M, N, K, scale, iter);

    LOGI("[ test_fpde_mini_matmul4x4 ] start...\n");
    test_fpde_mini_matmul4x4();
    LOGI("[ test_fpde_mini_matmul4x4 ] end...\n\n");

    LOGI("[ test_fpde_mini_matmul_quantize4x4 ] start...\n");
    test_fpde_mini_matmul_quantize4x4();
    LOGI("[ test_fpde_mini_matmul_quantize4x4 ] end...\n\n");

    LOGI("[ test_fpde_mini_matmul4x3 ] start...\n");
    test_fpde_mini_matmul4x3();
    LOGI("[ test_fpde_mini_matmul4x3 ] end...\n\n");

    return 0;
}