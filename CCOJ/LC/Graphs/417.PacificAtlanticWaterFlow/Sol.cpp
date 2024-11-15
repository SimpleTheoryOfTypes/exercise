#include <vector>
#include <deque>
#include <utility>
#include <map>
using namespace std;

class Solution {
  map<pair<int, int>, bool> pacific;
  map<pair<int, int>, bool> atlantic;
  map<pair<int, int>, int> color;
  int nrows = 0;
  int ncols = 0;
public:
    vector<vector<int>> pacificAtlantic(const vector<vector<int>>& heights) {
      nrows = heights.size();
      ncols = heights[0].size();

      
      for (int r = 0; r < nrows; r++) {
        pacific[std::make_pair(r, 0)] = true;
        atlantic[std::make_pair(r, ncols - 1)] = true;
      }

      for (int c = 0; c < ncols; c++) {
        pacific[std::make_pair(0, c)] = true;
        atlantic[std::make_pair(nrows - 1, c)] = true;
      }
     

      vector<vector<int>> ans;
      for (int r = 0; r < nrows; r++)  {
        for (int c = 0; c < ncols; c++) {
          color.clear();
          bfs(r, c, heights);
          if (pacific.find(std::make_pair(r,c)) != pacific.end() &&
              atlantic.find(std::make_pair(r,c)) != atlantic.end())
              ans.push_back({r,c});
        }
      }

      return ans;
    }

    void bfs(int row, int col, const vector<vector<int>>& heights) {
      std::vector<int> directions = {-1, 1};
      std::deque<vector<int>> Q;
      Q.push_back({row, col, heights[row][col]});
      color[std::make_pair(row,col)] = 1;
      while (!Q.empty()) {
        auto item = Q.front();
        auto r = item[0];
        auto c = item[1];
        auto u = item[2];
        Q.pop_front();

        // Early exit.
        if (pacific.find(std::make_pair(r,c)) != pacific.end() &&
            atlantic.find(std::make_pair(r,c)) != atlantic.end()) {
              pacific[std::make_pair(row, col)] = true;
              atlantic[std::make_pair(row, col)] = true;
              return;
        }

        for (auto d : directions) {
          if (r + d >= 0 && r + d < nrows && u >= heights[r+d][c] && color.find(std::make_pair(r+d,c)) == color.end()) {
            if (r+d == 0)
              pacific[std::make_pair(row, col)] = true;
            if (r+d == (nrows - 1))
              atlantic[std::make_pair(row, col)] = true;
            Q.push_back({r+d, c, heights[r+d][c]});
            color[std::make_pair(r+d,c)] = 1;
          }

          if (c + d >= 0 && c + d < ncols && u >= heights[r][c+d] && color.find(std::make_pair(r,c+d)) == color.end()) {
            if (c+d == 0)
              pacific[std::make_pair(row,col)] = true;
            if (c+d == ncols - 1)
              atlantic[std::make_pair(row,col)] = true;
            Q.push_back({r, c+d, heights[r][c+d]});
            color[std::make_pair(r,c+d)] = 1;
          }
        }
      }
    }
};

int main() {
  auto sol = Solution();
  auto ans = sol.pacificAtlantic({{1,2,2,3,5},{3,2,3,4,4},{2,4,5,3,1},{6,7,1,4,5},{5,1,1,2,4}});
  return ans.size();
}
