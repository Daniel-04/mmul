all:
	gcc -fopenmp -lopenblas -O3 bench.c mmul.c

debug:
	gcc -fopenmp -lopenblas -fsanitize=address,undefined,leak,bounds-strict -g bench.c mmul.c
