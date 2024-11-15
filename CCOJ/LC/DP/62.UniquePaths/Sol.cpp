#include <iostream>
#include <vector>
using namespace std;

class Solution {
public:
    int uniquePaths(int m, int n) {
      vector<vector<int>> board(m, vector<int>(n));

      // Don't forget corner cases!!
      if (m == 1 || n ==1)
        return 1;


      // Let f(i, j) denote the number of unique paths from (0,0) to (i,j)
      // Then, f(i,j) = f(i-1,j) + f(i,j-1)
      // Also, initial conditions: f(0,0) = 0, f(i=0,j) = 1 and f(i,j=0) = 1.
      for (int row = 0; row < m; row++)
        board[row][0] = 1;

      for (int col = 0; col < n; col++)
        board[0][col] = 1;

      board[0][0] = 0;

      for (int row = 1; row < m; row++)
        for (int col = 1; col < n; col++)
          board[row][col] = board[row-1][col] + board[row][col-1];
        
      return board[m-1][n-1];
    }
};

int main() {
  auto sol = Solution();
  auto ans = sol.uniquePaths(3,7);
  return ans;
}
