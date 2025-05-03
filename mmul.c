#include "mmul.h"
#include <stdlib.h>

void basic_mmul(int M1, int N2, int N1, const float *A, const float *B,
                float *C) {
    for (int i = 0; i < M1; i++) {
        for (int j = 0; j < N2; j++) {
            C[i * N2 + j] = 0;
            for (int k = 0; k < N1; k++) {
                C[i * N2 + j] += A[i * N1 + k] * B[k * N2 + j];
            }
        }
    }
}

void restrict_mmul(int M1, int N2, int N1, const float *restrict A,
                   const float *restrict B, float *restrict C) {
    for (int i = 0; i < M1; i++) {
        for (int j = 0; j < N2; j++) {
            C[i * N2 + j] = 0;
            for (int k = 0; k < N1; k++) {
                C[i * N2 + j] += A[i * N1 + k] * B[k * N2 + j];
            }
        }
    }
}

#define TILE_SIZE 16
void tiled_mmul(int M1, int N2, int N1, const float *restrict A,
                const float *restrict B, float *restrict C) {
    for (int it = 0; it < M1; it += TILE_SIZE) {
        for (int jt = 0; jt < N2; jt += TILE_SIZE) {
            for (int kt = 0; kt < N1; kt += TILE_SIZE) {
                for (int i = it; i < it + TILE_SIZE && i < M1; i++) {
                    for (int j = jt; j < jt + TILE_SIZE && j < N2; j++) {
                        float sum = 0;
                        for (int k = kt; k < kt + TILE_SIZE && k < N1; k++) {
                            sum += A[i * N1 + k] * B[k * N2 + j];
                        }
                        C[i * N2 + j] += sum;
                    }
                }
            }
        }
    }
}

void openmp_mmul(int M1, int N2, int N1, const float *restrict A,
                 const float *restrict B, float *restrict C) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < M1; i++) {
        for (int j = 0; j < N2; j++) {
            C[i * N2 + j] = 0;
            for (int k = 0; k < N1; k++) {
                C[i * N2 + j] += A[i * N1 + k] * B[k * N2 + j];
            }
        }
    }
}

void transposed_mmul(int M1, int N2, int N1, const float *restrict A,
                     const float *restrict B, float *restrict C) {
    float *restrict _B = malloc(sizeof(float) * N2 * N1);
    if (!_B)
        return;

    for (int i = 0; i < N1; i++) {
        for (int j = 0; j < N2; j++) {
            _B[j * N1 + i] = B[i * N2 + j];
        }
    }

    for (int i = 0; i < M1; i++) {
        for (int j = 0; j < N2; j++) {
            C[i * N2 + j] = 0;
            for (int k = 0; k < N1; k++) {
                C[i * N2 + j] += A[i * N1 + k] * _B[j * N1 + k];
            }
        }
    }

    free(_B);
}

void tiled_transposed_mmul(int M1, int N2, int N1, const float *restrict A,
                           const float *restrict B, float *restrict C) {
    float *restrict _B = malloc(sizeof(float) * N2 * N1);
    if (!_B)
        return;

    // Initialise C to zero
    for (int i = 0; i < M1; i++) {
        for (int j = 0; j < N2; j++) {
            C[i * N2 + j] = 0;
        }
    }

    // Transpose B into _B
    for (int i = 0; i < N1; i++) {
        for (int j = 0; j < N2; j++) {
            _B[j * N1 + i] = B[i * N2 + j];
        }
    }

    for (int it = 0; it < M1; it += TILE_SIZE) {
        for (int jt = 0; jt < N2; jt += TILE_SIZE) {
            for (int kt = 0; kt < N1; kt += TILE_SIZE) {
                for (int i = it; i < it + TILE_SIZE && i < M1; i++) {
                    for (int j = jt; j < jt + TILE_SIZE && j < N2; j++) {
                        float sum = 0;
                        for (int k = kt; k < kt + TILE_SIZE && k < N1; k++) {
                            sum += A[i * N1 + k] * _B[j * N1 + k];
                        }
                        C[i * N2 + j] += sum;
                    }
                }
            }
        }
    }

    free(_B);
}

void openmp_tiled_transposed_mmul(int M1, int N2, int N1,
                                  const float *restrict A,
                                  const float *restrict B, float *restrict C) {
    float *restrict _B = malloc(sizeof(float) * N2 * N1);
    if (!_B)
        return;

    // Initialise C to zero
#pragma omp parallel for collapse(2)
    for (int i = 0; i < M1; i++) {
        for (int j = 0; j < N2; j++) {
            C[i * N2 + j] = 0;
        }
    }

    // Transpose B into _B
#pragma omp parallel for collapse(2)
    for (int i = 0; i < N1; i++) {
        for (int j = 0; j < N2; j++) {
            _B[j * N1 + i] = B[i * N2 + j];
        }
    }

#pragma omp parallel for collapse(2)
    for (int it = 0; it < M1; it += TILE_SIZE) {
        for (int jt = 0; jt < N2; jt += TILE_SIZE) {
            for (int kt = 0; kt < N1; kt += TILE_SIZE) {
                for (int i = it; i < it + TILE_SIZE && i < M1; i++) {
                    for (int j = jt; j < jt + TILE_SIZE && j < N2; j++) {
                        float sum = 0;
                        for (int k = kt; k < kt + TILE_SIZE && k < N1; k++) {
                            sum += A[i * N1 + k] * _B[j * N1 + k];
                        }
                        C[i * N2 + j] += sum;
                    }
                }
            }
        }
    }

    free(_B);
}
