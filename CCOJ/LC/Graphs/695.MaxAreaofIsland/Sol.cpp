#include <vector>
using namespace std;

class Solution {
  public:
    vector<int> areas;// areas of islands
    int nrows;
    int ncols;

    int maxAreaOfIsland(const vector<vector<int>>& grid) {
      if (grid.empty())
        return 0;

      nrows = grid.size();
      ncols = grid[0].size();

      if (sum2D(grid) <= 0)
        return 0;

      if (sum2D(grid) == 1)
        return 1;

      vector<vector<int>> color(nrows, vector<int>(ncols, 0));
      dfs(grid, color);
                  
      return *max_element(areas.begin(), areas.end());
    }

    int sum2D(const vector<vector<int>>& grid) {
      int sum = 0;
      for (int i = 0; i < nrows; i++)
        for (int j = 0; j < ncols; j++)
          sum += grid[i][j];
      return sum;
    }

    void dfs(const vector<vector<int>>& grid, vector<vector<int>> &color) {
      for (int r = 0; r < nrows; r++) {
        for (int c = 0; c < ncols; c++) {
          if (color[r][c] == 0 && grid[r][c] == 1) {
            int area = 0;
            dfs_visit(grid, color, r, c, area);
            areas.push_back(area);
          }
        }
      } 
    }

    void dfs_visit(const vector<vector<int>>& grid, vector<vector<int>>& color, const int r, const int c, int &area) {
      if (grid[r][c] == 1) {
        area += 1;
        color[r][c] = 1;
      }

      if (inBound(r+1, nrows) && inBound(c, ncols) && color[r+1][c] == 0 && grid[r+1][c] == 1)
        dfs_visit(grid, color, r+1, c, area);

      if (inBound(r, nrows) && inBound(c+1, ncols) && color[r][c+1] == 0 && grid[r][c+1] == 1)
        dfs_visit(grid, color, r, c+1, area);
      
      if (inBound(r-1, nrows) && inBound(c, ncols) && color[r-1][c] == 0 && grid[r-1][c] == 1)
        dfs_visit(grid, color, r-1, c, area);

      if (inBound(r, nrows) && inBound(c-1, ncols) && color[r][c-1] == 0 && grid[r][c-1] == 1)
        dfs_visit(grid, color, r, c-1, area);
    }

    bool inBound(int x, int n) {
      if (x >= 0 && x < n)
        return true;
      return false;
    }
};

int main() {
  auto sol = Solution();
  auto ans = sol.maxAreaOfIsland(
      {{0,0,1,0,0,0,0,1,0,0,0,0,0},
       {0,0,0,0,0,0,0,1,1,1,0,0,0},
       {0,1,1,0,1,0,0,0,0,0,0,0,0},
       {0,1,0,0,1,1,0,0,1,0,1,0,0},
       {0,1,0,0,1,1,0,0,1,1,1,0,0},
       {0,0,0,0,0,0,0,0,0,0,1,0,0},
       {0,0,0,0,0,0,0,1,1,1,0,0,0},
       {0,0,0,0,0,0,0,1,1,0,0,0,0}});
  return ans;
}
