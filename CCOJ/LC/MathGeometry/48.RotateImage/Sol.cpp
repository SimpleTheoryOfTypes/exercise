#include <vector>
#include <iostream>
#include <iomanip>
using namespace std;

class Solution {
  public:
    void rotate(vector<vector<int>>& matrix) {

      int n = matrix.size();
      if (n == 0 || n == 1)
        return;

      printVector(matrix);
      assert(matrix[0].size() == n);
      int half_n = (n & 1) ? n/2+1 : n/2;
      int count = 0;
      bool nIsOdd = (n & 1);
      int rowBound = nIsOdd ? half_n-1 : half_n;
      // When image size is  odd, say 5, we only need to rotate elements in the region of matrix[0:1, 0:2] 
      // When image size is even, say 6, we only need to rotate elements in the region of matrix[0:2, 0:2] 
      for (int r = 0; r < rowBound; r++) {
        for (int c = 0; c < half_n; c++) {
          std::cout << "(SimpleTheoryOfTypes) (r,c) = " << r << "," << c << std::endl;
          int temp = matrix[r][c];
          //matrix[r][c] = matrix[c][n-r-1];
          //matrix[c][n-r-1] = matrix[n-r-1][n-c-1];
          //matrix[n-r-1][n-c-1] = matrix[n-c-1][r];
          //matrix[n-c-1][r] = temp;
          matrix[r][c] = matrix[n-c-1][r];
          matrix[n-c-1][r] = matrix[n-r-1][n-c-1];
          matrix[n-r-1][n-c-1] = matrix[c][n-r-1];
          matrix[c][n-r-1] = temp;
          printVector(matrix);
        }
        count += 1;
      }
    }

    void printVector(vector<vector<int>>& M) {
      for (auto &m : M) {
        std::cout << "\n";
        for (auto &x : m)
          std::cout << setw(2) << x << ",";
      }
      std::cout << "\n";
    }
};

int main() {
  auto sol = Solution();
  /* vector<vector<int>> matrix({{1,2,3},{4,5,6},{7,8,9}}); */
  /* vector<vector<int>> matrix({{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}}); */
  vector<vector<int>> matrix({{1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15},{16,17,18,19,20},{21,22,23,24,25}});
  sol.rotate(matrix);
  return 0;
}
