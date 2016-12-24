EULLVM38=${HOME}/dev/LLVM/llvm-3.8.0.build
SYS_CC=/usr/bin/gcc
CC=clang
INCLUDE_FLAGS=${EULLVM38}/projects/openmp/runtime/src
LIB_FLAGS=${EULLVM38}/lib
SYS_INCLUDE=
all:main main-sys
main:test_llvm_openmp.c
	LD_LIBRARY_PATH=${LIB_FLAGS} ${CC} -fopenmp -O0 -o $@ $^ -Wall -I ${INCLUDE_FLAGS} -L ${LIB_FLAGS} -lm
main-sys:test_llvm_openmp.c
	${SYS_CC} -std=gnu99 -O0 -o $@ $^ -I{SYS_INCLUDE_FLAGS} -fopenmp -Wall -lm
main.s:test_llvm_open.c
	${CC} -S -o $@ $^ -Wall -fopenmp -lm
test:
	time ./main
	time ./main-sys


.PHONY:clean
clean:
	rm -f *.s *.ll .*.swp main*
