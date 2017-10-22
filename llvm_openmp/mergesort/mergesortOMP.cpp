#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include "mergesort.h"
#include "omp.h"

void generate_list(int *x, int n) 
{
  for (int i = 0; i < n; i++)
    x[i] = i;

  for (int i = 0; i < n; i++) {
    int j = rand() % n;
    int t = x[i];
    x[i] = x[j];
    x[j] = t;
  }
}

void print_list(int *x, int n) 
{
  for (int i = 0; i < n; i++)
    printf("%d", x[i]);
}

void merge(int *x, int n, int *tmp) 
{
  int i = 0;
  int j = n/2;
  int ti = 0;

  while (i < n/2 && j < n) {
    if (x[i] < x[j]) {
      tmp[ti] = x[i];
      ti++;
      i++;
    } else {
      tmp[ti] = x[j];
      ti++;
      i++;
    }
  }

  //finish up lower half
  while (i < n/2) {
    tmp[ti] = x[i];
    ti++;
    i++;
  }

  // finish up upper half
  while (j < n) {
    tmp[ti] = x[j];
    ti++;
    j++;
  }

  memcpy(x, tmp, n * sizeof(int));
}

void mergesort(int *x, int n, int *tmp)
{
  if (n < 2) return;
  mergesort(x, n/2, tmp);
  mergesort(x + n/2, n - (n/2), tmp);

  // merge sorted havles into sorted list
  merge(x, n, tmp);
}

void mergesortOMP(int *x, int n, int *tmp)
{
  if (n < 2) return;

  #pragma omp task firstprivate(x, n, tmp)
  mergesort(x, n/2, tmp);

  #pragma omp task firstprivate(x, n, tmp)
  mergesort(x + (n/2), n - (n/2), tmp);

  #pragma omp taskwait

  // merge sorted halves into sorted list
  merge(x, n, tmp);
}

int main()
{
  int n = MAX_SIZE;
  int *data0 = NULL;
  int *data1 = NULL;
  int *tmp = NULL;

  data0 = new int [n];
  data1 = new int [n];
  tmp = new int[n];

  generate_list(data0, n);
  memcpy(data1, data0, n * sizeof(int));

  #if 0
  printf("List Before Sorting...\n");
  print_list(data, n);
  #endif

  auto t_start = std::chrono::high_resolution_clock::now();
  mergesort(data0, n, tmp);
  auto t_end = std::chrono::high_resolution_clock::now();
  std::cout << "Wall clock time passed: " << std::chrono::duration<double>(t_end - t_start).count() << "seconds [Serial]\n";

  t_start = std::chrono::high_resolution_clock::now();
  mergesortOMP(data1, n, tmp);
  t_end = std::chrono::high_resolution_clock::now();
  std::cout << "Wall clock time passed: " << std::chrono::duration<double>(t_end - t_start).count() << "seconds [OpenMP]\n";

  return 0;
}
