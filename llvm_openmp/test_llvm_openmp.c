#include <stdio.h>
#include <math.h>
#include <omp.h>

int main()
{
  #ifdef _OPENMP
  printf("Compiled by an OpenMP-compliant implementation\n");
  #endif

  int x = 2;
  #pragma omp parallel num_threads(2) shared(x)
  {
    if (omp_get_thread_num() == 0) {
      printf("0: Thread#: %d: x = %d\n", omp_get_thread_num(), x);
      x = 5;
	} else {
      printf("1: Thread#: %d: x = %d\n", omp_get_thread_num(), x);
	}

    #pragma omp barrier
	if (omp_get_thread_num() == 0) {
      printf("2: Thread#: %d: x = %d\n", omp_get_thread_num(), x);
	} else {
      printf("3: Thread#: %d: x = %d\n", omp_get_thread_num(), x);
	}
  }

  #if _OPENMP
  int N = 999999999;
  double sum = 0.0;
  #pragma omp parallel for num_threads(8)
  for (int i = 0; i < N; ++i) {
    double y = 0.0;
	y = (log2(i + 1));
	sum += y;
  }
  #endif
  
  return (int) sum;
}
