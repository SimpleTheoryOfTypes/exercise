#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <cstdio>

struct _ATPair {
  double val;
  double clas;
};

typedef std::vector<struct _ATPair> ATPairT;

struct CompAttrTargetPairObj {
  bool operator() (const struct _ATPair &a, const struct _ATPair &b) {
    return (a.val < b.val);
  }
} CompAttrTargetPairObj;

int compare(const void *a, const void *b) {
  return (*(int*) a - *(int*) b);
}

void profile_std_sort(int num_data)
{
  ATPairT vcv;
  for (int example = 0; example < num_data; ++example) {
    struct _ATPair atp;
	atp.val = rand();
	atp.clas = example;
	vcv.push_back(atp);
  }

  std::sort(vcv.begin(), vcv.end(), CompAttrTargetPairObj);
}

void profile_std_qsort(int num_data)
{
  double *p = new double [num_data];
  for (int example = 0; example < num_data; ++example) {
    p[example] = rand();
  }
  std::qsort((void *)p, num_data, sizeof(double), compare);
  delete [] p;
}

int main(int argc, char *argv[])
{
  if (argc != 2) {
    printf("Usage: ./test_std_sort_qsort q\n");  
	exit(1);
  }

  int num_data = 199324970;
  if (argv[1][0] == 'q')
	profile_std_qsort(num_data);
  else 
    profile_std_sort(num_data);

  return 0;
}
