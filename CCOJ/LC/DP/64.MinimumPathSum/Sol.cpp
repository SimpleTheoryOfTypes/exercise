#include <vector>
using namespace std;

class Solution {
  public:
    int minPathSum(const vector<vector<int>>& grid) {
      if (grid.empty())
        return 0;

      int nrows = grid.size();
      int ncols = grid[0].size();
      vector<vector<int>> dp(nrows, vector<int>(ncols, 0));

      dp[0][0] = grid[0][0];
      for (auto r = 1; r < nrows; r++)
        dp[r][0] = dp[r-1][0] + grid[r][0];
      for (auto c = 1; c < ncols; c++)
        dp[0][c] = dp[0][c-1] + grid[0][c];


      for (auto r = 1; r < nrows; r++)
        for (auto c = 1; c < ncols; c++)
          dp[r][c] = min(dp[r-1][c], dp[r][c-1]) + grid[r][c];

      return dp[nrows-1][ncols-1];
    }
};

int main() {
  auto sol = Solution();
  auto ans = sol.minPathSum({{1,3,1},{1,5,1},{4,2,1}});
  return ans;
}
