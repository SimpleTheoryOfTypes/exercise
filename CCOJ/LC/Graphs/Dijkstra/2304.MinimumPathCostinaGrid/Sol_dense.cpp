#include <vector>
using namespace std;

class Solution {
public:
    int minPathCost(const vector<vector<int>>& grid, const vector<vector<int>>& moveCost) {
      int nrows = grid.size();
      int ncols = grid[0].size();
      int n = nrows * ncols;// total num of nodes.
      auto g = vector<vector<pair<int, int>>>(n, vector<pair<int, int>>());
      for (int r = 0; r < nrows - 1; r++) {
        for (int c1 = 0; c1 < ncols; c1++) {
          int from = grid[r][c1];
          for (int c2 = 0; c2 < ncols; c2++) {
            int to = grid[r+1][c2];
            g[from].push_back({to, from + moveCost[from][c2]});
          }
        }
      }

      int ans = INT_MAX;
      for (int c = 0; c < ncols; c++) {
        int s = grid[0][c];

        vector<int> d(n, INT_MAX);
        d[s] = 0;
        vector<bool> mark(n, false);
        vector<int> p(n, -1);

        for (int i = 0; i < n; i++) {
          int v = -1;
          for (int j = 0; j < n; j++) {
            if (!mark[j] && (v == -1 || d[j] < d[v]))
              v = j;
          }

          if (d[v] == INT_MAX)
            break;

          mark[v] = true;
          for (const auto &[nbr, wt]: g[v]) {
            if (d[v] + wt < d[nbr]) {
              d[nbr] = d[v] + wt;
              p[v] = nbr;
            }
          }
        }

        for (int c1 = 0; c1 < ncols; c1++) {
          int n = grid[nrows-1][c1];
          ans = min(ans, d[n] + n);
        }
      }

      return ans;
    }
};

int main() {
  auto sol = Solution();
  auto ans = sol.minPathCost({{5,3},{4,0},{2,1}}, {{9,8},{1,5},{10,12},{18,6},{2,4},{14,3}});
  return ans;
}

