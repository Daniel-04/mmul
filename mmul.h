#ifndef MMUL_H_
#define MMUL_H_

void basic_mmul(int M1, int N2, int N1, const float *A, const float *B,
                float *C);

void restrict_mmul(int M1, int N2, int N1, const float *A, const float *B,
                   float *C);

void tiled_mmul(int M1, int N2, int N1, const float *A, const float *B,
                float *C);

void openmp_mmul(int M1, int N2, int N1, const float *A, const float *B,
                 float *C);

void transposed_mmul(int M1, int N2, int N1, const float *A, const float *B,
                     float *C);

void tiled_transposed_mmul(int M1, int N2, int N1, const float *A,
                           const float *B, float *C);

void openmp_tiled_transposed_mmul(int M1, int N2, int N1, const float *A,
                                  const float *B, float *C);

#endif // MMUL_H_
