all:
	gcc -fopenmp -lopenblas -O3 bench.c mmul.c

debug:
	gcc -fopenmp -lopenblas -O3 -fsanitize=address,undefined -g bench.c mmul.c
