#include <vector>
#include <deque>
using namespace std;

//https://www.youtube.com/watch?v=J_o6XVRXuHs
class Solution {
  int nrows;
  int ncols;
public:
    int shortestPathBinaryMatrix(const vector<vector<int>>& grid) {
      nrows = grid.size();
      ncols = grid[0].size();

      if (grid[0][0] == 1 || grid[nrows-1][ncols-1] == 1)
        return -1;

      vector<vector<int>> dist(nrows, vector<int>(ncols, INT_MAX));
      dist[0][0] = 0;


      vector<vector<int>> directions = {{0,1}, {0,-1}, {1,0}, {-1,0}, {1,1}, {1,-1},{-1,1},{-1,-1}};
      deque<vector<int>> q;
      q.push_back({0, 0});

      while (!q.empty()) {
        auto u = q.front();
        int r = u[0];
        int c = u[1];
        q.pop_front();

        for (const auto &d : directions) {
          int dr = d[0];
          int dc = d[1];
          int r2 = r + dr;
          int c2 = c + dc;
          if (r2 >= 0 && r2 < nrows && c2 >= 0 && c2 < ncols && grid[r2][c2] == 0 && dist[r2][c2] == INT_MAX) {
            dist[r2][c2] = dist[r][c] + 1;
            q.push_back({r2,c2});
          }
        }
      }

      if (dist[nrows - 1][ncols - 1] == INT_MAX)
        return -1;

      return dist[nrows - 1][ncols - 1] + 1;
    }
};


int main() {
  auto sol = Solution();
  auto ans = sol.shortestPathBinaryMatrix({{1,0,0},{1,1,0},{1,1,0}});
  return ans;
}
