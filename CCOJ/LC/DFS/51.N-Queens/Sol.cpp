#include <vector>
#include <iostream>
#include <string>
#include <map>
#include <utility>
using namespace std;

class Solution {
    vector<vector<string>> ans;
    map<pair<int, int>, int> occupied;
    int numPlaced = 0;
  public:
    vector<vector<string>> solveNQueens(int n) {
      if (n == 0)
        return {};
      if (n == 1)
        return {{"Q"}};

      string x(n, '.');// Repeat '.' n times.
      vector<string> board (n, x);
      backtrack(board, 0, n);
      return ans;
    }

    bool isValidPlacement(int r, int c, bool skipSelf=true) {
      for (const auto &[pr, ignore] : occupied) {
        int row = pr.first;
        int col = pr.second;

        if (r == row && c == col && skipSelf == true)
          continue;

        if (row == r || col == c)
          return false;

        // diagonal: y = x + b or y = -x + b
        // I.e., r - c = b is a constant or r + c = b is a constant.
        if ((r - c == row - col) || (r + c == row + col)) {
          return false;
        }
      }

      return true;
    }

    bool isValidBoard() {
      for (const auto &[pr, ignore] : occupied) {
        if (!isValidPlacement(pr.first, pr.second))
          return false;
      }
      return true;
    }

    bool alreadyOccupied(int r, int c) {
      for (const auto &[pr, ignore] : occupied)
        if (pr.first == r || pr.second == c)
          return true;

      return false;
    }

    bool backtrack(vector<string> &board, int r, const int n) {
      if (!isValidBoard())
        return false;

      if (r == n)
        return true;

      for (int c = 0; c < n; c++) {
        if (board[r][c] == '.' && !alreadyOccupied(r, c) && isValidPlacement(r, c)) {
          board[r][c] = 'Q';
          occupied[std::make_pair(r, c)] = 1;
          numPlaced += 1;
          bool flag = backtrack(board, r+1, n);
          if (flag && numPlaced == n) {
            ans.push_back(board);
            printBoard(board);
          }
          board[r][c] = '.';
          occupied.erase(std::make_pair(r, c));
          numPlaced -= 1;
        }
      }

      return true;
    }

    void printBoard(vector<string> &board) {
      for (const auto &s : board) {
        std::cout << s << std::endl;
      }
      std::cout << "=======" << occupied.size() << "=====\n";
    }
};

int main() {
  auto sol = Solution();
  auto ans = sol.solveNQueens(8);
  return 0;
}
