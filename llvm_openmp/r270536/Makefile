INCLUDE_FLAGS=${HOME}/dev/LLVM/llvm.git.build/projects/openmp/runtime/src/
LIB_FLAGS=${HOME}/dev/LLVM/llvm.git.build/lib
all:r270536
r270536:r270536.cpp
	clang++ -Wall -o $@ $^ -fopenmp -I ${INCLUDE_FLAGS} -L ${LIB_FLAGS}
.PHONY:clean
clean:
	rm -f r270536 .*.sw*
