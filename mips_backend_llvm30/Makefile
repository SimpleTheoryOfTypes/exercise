all:sample.mips
sample.mips:sample.ll
	${EULLVM30}/llc -march=mipsel -o $@ $^
sample.ll:sample.c
	${EULLVM30}/clang -S -emit-llvm -O3 -o $@ $^

.PHONY:clean
clean:
	rm -f *.mips *.ll .*.swp
