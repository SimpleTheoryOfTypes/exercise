#include <omp.h>
#include <cstdio>
#include <cmath>
#include <unistd.h>

float foo()
{
  int i, chunk;
  unsigned int n = 134217728;
  double *a, *b, result, resultOMP;
  a = new double[n];
  b = new double[n];

  chunk = 1000000;
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    a[i] = log10((i + 1) * 0.3);
    b[i] = log10((i + 1) * 3.0);
  }

  resultOMP = 0.0;
  #pragma omp parallel for default(shared) private(i) schedule(static, chunk) reduction(+:resultOMP)
  for (i = 0; i < n; i++) {
    resultOMP = resultOMP + (a[i] * b[i]);
  }

  result = 0.0;
  for (i = 0; i < n; i++) {
    result = result + (a[i] * b[i]);
  }

  if (fabs(result - resultOMP) / fabs(result) < 0.001)
    printf("SUCCESS. [Serial %f] [OMP %f]\n", result, resultOMP);

  delete [] a;
  delete [] b;

  return resultOMP;
}

int main()
{
  float TheResult = foo();
  printf("[Result] %f\n", TheResult);
  return 0;
}
