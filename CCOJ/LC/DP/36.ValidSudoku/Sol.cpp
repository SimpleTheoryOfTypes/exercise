#include <vector>
#include <map>
using namespace std;

class Solution {
public:
    bool isValidSudoku(const vector<vector<char>>& board) {
      map<char, int> hashMap;
      int nrows = board.size();
      int ncols = board[0].size();

      for (int i = 0; i < nrows; i++) {
        const auto &v = board[i];
        hashMap.clear();
        for (const auto &x : v) {
          if (x != '.' && hashMap.find(x) != hashMap.end())
            return false;
          hashMap[x] = 1;
        }
      }

      for (int c = 0; c < ncols; c++) {
        hashMap.clear();
        for (int r = 0; r < nrows; r++) {
          if (board[r][c] != '.' && hashMap.find(board[r][c]) != hashMap.end())
            return false;
          hashMap[board[r][c]] = 1;
        }
      }

      for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
          hashMap.clear();
          for (int rr = r*3; rr < (r+1)*3; rr++) {
            for (int cc = c*3; cc < (c+1)*3; cc++) {
              if (board[rr][cc] != '.' && hashMap.find(board[rr][cc]) != hashMap.end())
                return false;
              hashMap[board[rr][cc]] = 1;
            }
          }
        }
      }

      return true;
    }
};

int main() {
  auto sol = Solution();
  auto ans = sol.isValidSudoku({{'5','3','.','.','7','.','.','.','.'}
                               ,{'6','.','.','1','9','5','.','.','.'}
                               ,{'.','9','8','.','.','.','.','6','.'}
                               ,{'8','.','.','.','6','.','.','.','3'}
                               ,{'4','.','.','8','.','3','.','.','1'}
                               ,{'7','.','.','.','2','.','.','.','6'}
                               ,{'.','6','.','.','.','.','2','8','.'}
                               ,{'.','.','.','4','1','9','.','.','5'}
                               ,{'.','.','.','.','8','.','.','7','9'}});
  return ans;
}
