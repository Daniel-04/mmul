#include <openblas/cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "mmul.h"

#define M1 1024
#define N1 1024
#define M2 1024
#define N2 1024

/*
**      N1
**    A --
** M1 | ..
**    | ..
**
**      N2
**    B --
** M2 | ..
**    | ..
**
**      N2
**    C --
** M1 | ..
**    | ..
**
** M2 = N1
*/

int main() {
    if (M2 != N1) {
        fprintf(stderr, "Incompatible matrix dimensions.\n");
        return -1;
    }

    float *A = malloc(sizeof(float) * M1 * N1);
    float *B = malloc(sizeof(float) * M2 * N2);
    float *C = malloc(sizeof(float) * M1 * N2);
    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed.\n");
        return 1;
    }

    for (int i = 0; i < M1 * N1; i++)
        A[i] = (float)(i % 100) / 10.0f;
    for (int i = 0; i < M2 * N2; i++)
        B[i] = (float)(i % 100) / 10.0f;

    printf("\t%d\n"
           "%d\tA\n"
           "\n"
           "\t%d\n"
           "%d\tB\n"
           "\n"
           "\t%d\n"
           "%d\tC\n",
           N1, M1, N2, M2, N2, M1);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M1, N2, N1, 1.0f, A,
                N1, B, N2, 0.0f, C, N2);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed =
        (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("%.6f\tseconds taken for matrix multiplication (sgemm)\n", elapsed);

    clock_gettime(CLOCK_MONOTONIC, &start);
    basic_mmul(M1, N2, N1, A, B, C);
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("%.6f\tseconds taken for matrix multiplication (basic)\n", elapsed);

    clock_gettime(CLOCK_MONOTONIC, &start);
    restrict_mmul(M1, N2, N1, A, B, C);
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("%.6f\tseconds taken for matrix multiplication (restrict)\n",
           elapsed);

    clock_gettime(CLOCK_MONOTONIC, &start);
    tiled_mmul(M1, N2, N1, A, B, C);
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("%.6f\tseconds taken for matrix multiplication (tiled)\n", elapsed);

    clock_gettime(CLOCK_MONOTONIC, &start);
    openmp_mmul(M1, N2, N1, A, B, C);
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("%.6f\tseconds taken for matrix multiplication (openmp)\n", elapsed);

    clock_gettime(CLOCK_MONOTONIC, &start);
    transposed_mmul(M1, N2, N1, A, B, C);
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("%.6f\tseconds taken for matrix multiplication (transposed)\n",
           elapsed);

    clock_gettime(CLOCK_MONOTONIC, &start);
    tiled_transposed_mmul(M1, N2, N1, A, B, C);
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("%.6f\tseconds taken for matrix multiplication (tiled and "
           "transposed)\n",
           elapsed);

    clock_gettime(CLOCK_MONOTONIC, &start);
    openmp_tiled_transposed_mmul(M1, N2, N1, A, B, C);
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("%.6f\tseconds taken for matrix multiplication (tiled and "
           "transposed with openmp)\n",
           elapsed);

    free(A);
    free(B);
    free(C);

    return 0;
}

/*
** AMD Ryzen 9 5900X:
** 12 Cores, 4.8 GHz, AVX2 (8 mul, 8 add) ~16
** 12 * 4.8 * 16 = 921.6 Gflops
** ~= 921600000000 flops
**
** MMUL flops:
** 2 * M1 * N1 * N2
**
** 2 * 1024 * 1024 * 1024
** =    2147483648
**
** 2147483648 / 921600000000
** ~= 0.002330 seconds
** vs
**    0.040943 seconds (tiled and transposed with openmp)
**
** ~ 20x slower
*/
