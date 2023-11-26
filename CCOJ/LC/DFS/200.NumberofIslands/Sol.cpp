#include <vector>
#include <iostream>
using namespace std;

// DFS traverse grid once, mark every element '1' if
// it can be reached from a source node. Only start
// a new traversal from a unvisited element (i.e.,
// mask[element] = 0;
// The answer equals the number of new traversals. 
class Solution {
  int nrows;
  int ncols;
  vector<vector<int>> mask;
public:
    int numIslands(const vector<vector<char>>& grid) {
      nrows = grid.size();
      ncols = grid[0].size();
      mask.resize(nrows, vector<int>(ncols));

      int ans = 0;
        
      for (size_t i = 0; i < nrows; i++) {
        for (size_t j = 0; j < ncols; j++) {
          if (grid[i][j] == '1' && mask[i][j] == 0) {
            /* printVector(mask); */
            ans += 1;
            DFS_visit(grid, i, j);
            /* printVector(mask); */
          }
        }
      }

      return ans;
    }

    void printVector(const vector<vector<int>> &V) {
      for (auto v : V) {
        std::cout << std::endl;
        for (auto x : v) {
          std::cout << x << ",";
        }
        std::cout << std::endl;
      }
    }

    void DFS_visit(const vector<vector<char>> &grid, size_t i, size_t j) {
      mask[i][j] = 1;
      for (int y = -1; y <= 1; y++) {
        if (in_bound(j + y, 0, ncols) && grid[i][j + y] == '1' && mask[i][j + y] == 0) {
          DFS_visit(grid, i, j + y);
        }
      }
      for (int x = -1; x <= 1; x++) {
        if (in_bound(i + x, 0, nrows) && grid[i + x][j] == '1' && mask[i + x][j] == 0) {
          DFS_visit(grid, i + x, j);
        }
      }
    }

    bool in_bound(int i, int lb, int ub) {
      return (i >= lb) && (i < ub);
    }
};

int main() {
  auto sol = Solution();
  auto ans = sol.numIslands({
  {'1','1','1','1','0'},
  {'1','1','0','1','0'},
  {'1','1','0','0','0'},
  {'0','0','0','0','0'}
});

std::cout << "(SimpleTheoryOfTypes)" << ans << std::endl;
  return ans;
}
