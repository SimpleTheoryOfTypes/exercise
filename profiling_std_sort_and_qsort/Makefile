CC=clang++

all:test_std_sort_qsort test_std_sort_qsort.s test_std_sort_qsort.ll
test_std_sort_qsort: test_std_sort_qsort.cpp
	$(CC) -Wall -O3 -o $@ $^
test_std_sort_qsort.s: test_std_sort_qsort.cpp
	$(CC) -S -masm=intel -O3 -o $@ $^
test_std_sort_qsort.ll: test_std_sort_qsort.cpp
	$(CC) -S -emit-llvm -O3 -o $@ $^
	
.PHONY:clean
clean:
	rm -f test_std_sort_qsort *.s *.ll .*.swp
