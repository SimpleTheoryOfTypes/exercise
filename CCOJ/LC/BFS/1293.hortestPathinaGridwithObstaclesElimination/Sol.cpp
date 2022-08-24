#include <vector>
#include <queue>
#include <map>
using namespace std;

class Solution {
public:
    int shortestPath(const vector<vector<int>>& grid, int nobs) {
      int rows = grid.size();
      int cols = grid[0].size();

      map<vector<int>, int> dist;// map (r,c,#obs-so-far) to its distance to origin.
      dist[{0,0,0}] = 0;

      queue<vector<int>> q;
      q.push({0,0,0});
      vector<pair<int, int>> directions = {{0,1},{1,0},{0,-1},{-1,0}}; 

      while (!q.empty()) {
        const auto v = q.front();
        q.pop();
        int x = v[0], y = v[1], k = v[2];
        auto d = dist[{x,y,k}];
        for (const auto &[dx, dy] : directions) {
          int nx = x + dx;
          int ny = y + dy;

          if (nx >= 0 && nx < rows && ny >= 0 && ny < cols) {
            int nk = k + grid[nx][ny];
            if (nk <= nobs && dist.find({nx, ny, nk}) == dist.end()) {
              if (nx == rows - 1 && ny == cols - 1)
                return d + 1;

              dist[{nx, ny, nk}] = d + 1;
              q.push({nx, ny, nk});
            } 
          }
        }
      }
      return -1;
    }
};

int main() {
  auto sol = Solution();
  auto ans = sol.shortestPath({{}}, 1);
  return ans;
}
