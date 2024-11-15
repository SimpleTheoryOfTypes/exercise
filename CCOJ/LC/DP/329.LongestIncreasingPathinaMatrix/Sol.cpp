#include <vector>
using namespace std;

class Solution {
      // dp[r][c] denotes the longest increasing path starting at (r,c)
      vector<vector<int>> dp;
      int nrows;
      int ncols;
public:
    int longestIncreasingPath(const vector<vector<int>>& matrix) {
      if (matrix.empty())
        return 0;

      nrows = matrix.size();
      ncols = matrix[0].size();

      dp = vector<vector<int>>(nrows, vector<int>(ncols, -1));

      // Mark cells that are local max, and set its dp to 0;
      // Everything else is still -1 at this point.
      for (int r = 0; r < nrows; r++) {
        for (int c = 0; c < ncols; c++) {
          if ((inBound(r-1, nrows) && matrix[r][c] < matrix[r-1][c]) ||
              (inBound(r+1, nrows) && matrix[r][c] < matrix[r+1][c]) ||
              (inBound(c-1, ncols) && matrix[r][c] < matrix[r][c-1]) ||
              (inBound(c+1, ncols) && matrix[r][c] < matrix[r][c+1])) {
            continue;
          }
          // matrix[r][c] is local maxima among all its 4 neighors.
          dp[r][c] = 0;
        }
      }

      for (int r = 0; r < nrows; r++)
        for (int c = 0; c < ncols; c++)
          dfs(r,c,matrix);

      int ans = 0;
      for (int r = 0; r < nrows; r++)
        for (int c = 0; c < ncols; c++)
          ans = max(ans, dp[r][c]);

      return ans + 1;
    }

    bool inBound(int x, int b) {
      if (x >= 0 && x < b)
        return true;

      return false;
    }

    int dfs(int r, int c, const vector<vector<int>>& matrix) {
      if (dp[r][c] != -1)
        return dp[r][c];

      int myMax = -1;
      vector<int> Dir = {-1,1}; 
      for (const auto &i : Dir) {
        if (inBound(r+i, nrows) && matrix[r+i][c] > matrix[r][c] && dfs(r+i, c, matrix) > myMax)
          myMax = dfs(r+i,c, matrix);//FIXM: avoid calling the same func twice.
        if (inBound(c+i, ncols) && matrix[r][c+i] > matrix[r][c] && dfs(r, c+i, matrix) > myMax)
          myMax = dfs(r,c+i, matrix);
      }

      dp[r][c] = myMax + 1;
      return dp[r][c];
    }
};

int main() {
  auto sol = Solution();
  /* auto ans = sol.longestIncreasingPath({{9,9,4},{6,6,8},{2,1,1}}); */
  auto ans = sol.longestIncreasingPath({{1,2}});
  return ans;
}
