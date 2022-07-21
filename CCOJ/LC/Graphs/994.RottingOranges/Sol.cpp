#include <vector>
#include <algorithm>
#include <map>
#include <deque>
#include <utility>
using namespace std;

class Solution {
    map<pair<int, int>, int> color;
    map<pair<int, int>, int> depth;
    int nrows;
    int ncols;
public:
    int orangesRotting(const vector<vector<int>>& grid) {
      nrows = grid.size();
      ncols = grid[0].size();

      // Push all rotting oranges on Q, and do parallel BFS among them, recording the current breadth along the way.
      // The Final breadth is the anser.
      vector<int> discovers;
      deque<pair<int, int>> Q;
      for (int r = 0; r < nrows; r++) {
        for (int c = 0; c < ncols; c++) {
          if (color.find(make_pair(r,c)) == color.end() && grid[r][c] == 2) {
            Q.push_back(make_pair(r,c));
            depth[make_pair(r,c)] = 0;
          }
        }
      }

      int currDepth = 0;
      while (!Q.empty()) {
        bool flag = false;
        while (!Q.empty() && depth[Q.front()] == currDepth) {
          auto u = Q.front();
          Q.pop_front();
          int r = u.first;
          int c = u.second;

          if (inBound(r-1, c) && grid[r-1][c] == 1 && color.find(make_pair(r-1, c)) == color.end()) { Q.push_back(make_pair(r-1, c)); color[make_pair(r-1, c)] = 1; flag = true; depth[make_pair(r-1, c)] = currDepth + 1;}
          if (inBound(r+1, c) && grid[r+1][c] == 1 && color.find(make_pair(r+1, c)) == color.end()) { Q.push_back(make_pair(r+1, c)); color[make_pair(r+1, c)] = 1; flag = true; depth[make_pair(r+1, c)] = currDepth + 1;}
          if (inBound(r, c-1) && grid[r][c-1] == 1 && color.find(make_pair(r, c-1)) == color.end()) { Q.push_back(make_pair(r, c-1)); color[make_pair(r, c-1)] = 1; flag = true; depth[make_pair(r, c-1)] = currDepth + 1;}
          if (inBound(r, c+1) && grid[r][c+1] == 1 && color.find(make_pair(r, c+1)) == color.end()) { Q.push_back(make_pair(r, c+1)); color[make_pair(r, c+1)] = 1; flag = true; depth[make_pair(r, c+1)] = currDepth + 1;}
        }

        if (flag) currDepth += 1;
      }

      for (int r = 0; r < nrows; r++) {
        for (int c = 0; c < ncols; c++) {
          if (color.find(make_pair(r,c)) == color.end() && grid[r][c] == 1)
            return -1;
        }
      }

      return currDepth;
    }

    bool inBound(int r, int c) {
      return (r >= 0 && r < nrows && c >= 0 && c < ncols);
    }

};

int main() {
  auto sol = Solution();
  auto ans = sol.orangesRotting({{2,1,1},{0,1,1},{1,0,1}});
  /* auto ans = sol.orangesRotting({{0,2}}); */
  /* auto ans = sol.orangesRotting({{2,1,1},{1,1,0},{0,1,1}}); */
  return ans;
}
