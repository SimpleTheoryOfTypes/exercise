#include <cstdint>

// A c program that contains a loop that has the same dependence graph as in Fig.2.
// clang++ -S -emit-llvm -o test_fig2.ll test_fig2.cpp
void f(int *a, int *b, int *d, int *s, int *t, int c3, int c5, int c6, int64_t N) {
  int64_t i;
  for(i = 0; i < N; i++) {
    int n1 = a[i];
    int n3 = n1 + c3;
    int n5 = n1 + c5;
    int n2 = b[i];
    int n8 = n2 * n5;
    int n4 = n3 * n1;
    s[i] = n4; // n7
    int n9 = d[i];
    int n10 = n8 * n9;
    int n6 = n1 * c6;
    int n11 = n10 * n6;
    t[i] = n11; // n12
  }
}

