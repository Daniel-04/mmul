Sufficiently good C for naive matrix multiplication
Runtime for 1024x1024 A and B matrices in AMD Ryzen 9 5900X:

``` sh
$ make && ./a.out 
gcc -fopenmp -lopenblas -O3 bench.c mmul.c
        1024
1024    A

        1024
1024    B

        1024
1024    C
0.017302        seconds taken for matrix multiplication (sgemm)
3.804239        seconds taken for matrix multiplication (basic)
3.193697        seconds taken for matrix multiplication (restrict)
1.410646        seconds taken for matrix multiplication (tiled)
0.188158        seconds taken for matrix multiplication (openmp)
0.638646        seconds taken for matrix multiplication (transposed)
0.437068        seconds taken for matrix multiplication (tiled and transposed)
0.043164        seconds taken for matrix multiplication (tiled and transposed with openmp)
```

(12 Cores, 4.8 GHz, AVX2 (8 mul, 8 add) ~16)
12 * 4.8 * 16 = 921.6 Gflops
~= 921600000000 flops
                                                      
MMUL flops:
2 * M1 * N1 * N2
                                                      
2 * 1024 * 1024 * 1024
=    2147483648
                                                      
2147483648 / 921600000000 ~=
- 0.002330 seconds

vs
- 0.040943 seconds (tiled and transposed with openmp)
                                                      
~20x slower
