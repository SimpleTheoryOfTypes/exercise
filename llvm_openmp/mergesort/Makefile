CPP=clang++
INCLUDE_FLAGS=${LLVM_HOME}/projects/openmp/runtime/src
LIB_FLAGS=${LLVM_HOME}/lib
all:mergesortOMP
mergesortOMP:mergesortOMP.cpp
	LD_LIBRARY_PATH={LIB_FLAGS} ${CPP} --std=c++11 -fopenmp -o $@ $^ -Wall -I ${INCLUDE_FLAGS} -L ${LIB_FLAGS} -lm -O3
mergesortOMP.ll:mergesortOMP.cpp
	LD_LIBRARY_PATH={LIB_FLAGS} ${CPP} -S -emit-llvm --std=c++11 -fopenmp -o $@ $^ -Wall -I ${INCLUDE_FLAGS} -O3
clean:
	rm -f mergesortOMP *.ll
