#include <omp.h>
#include <stdio.h>
#include <math.h>

int main()
{
  int a = 2;
  #pragma omp parallel if (0)
  {
    #pragma omp for firstprivate(a)
    for (int i = 0; i < 1; i++) {
      printf("Hi from the first loop -- %d\n", a);
      a += 1;
    }

    #pragma omp for firstprivate(a)
    for (int i = 0; i < 1; i++) {
      printf("Hi from the second loop -- %d\n", a);
      a += 1;
    }
  }

  printf("Final -- %d\n", a);
  return 0;
}
