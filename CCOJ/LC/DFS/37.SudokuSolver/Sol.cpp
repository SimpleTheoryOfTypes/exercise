#include <vector>
#include <map>
#include <iostream>
using namespace std;

class Solution {
  int nrows;
  int ncols;
  bool Found = false;
  public:
    void solveSudoku(vector<vector<char>>& board) {
      if (board.empty())
        return;

      nrows = board.size();
      ncols = board[0].size();
      backtrack(board);
    }

    bool isValidSudoku(vector<vector<char>> &board, int row, int col, char c) {
      for(int i=0;i<9;i++) {
        if(board[i][col]==c)return false;
        if(board[row][i]==c)return false;
        if(board[3*(row/3)+i/3][3*(col/3)+i%3]==c)return false;
      }
      return true;
    }

    bool backtrack(vector<vector<char>> &board) {
      for (int r = 0; r < nrows; r++) {
        for (int c = 0; c < ncols; c++) {
          if (board[r][c] == '.') {
            for (int n = 1; n <= 9; n++) {
              if (isValidSudoku(board, r, c, n+'0')) {
                /* printSudoku(board); */
                board[r][c] = n + '0';
                if (backtrack(board)) return true;
                board[r][c] = '.';
              }
            }

            // This false is returned if none of the elements can be placed in
            // board[i][j](basically in the above loop),so it will return false
            // to the recursive call it got called, and so if it's false,
            // board[i][j] will again become "." This will not only return false,
            // it goes back the original recursive which called it and then
            // backtrack the value of the cell
            return false;
          }
        }
      }

      // If we have not entered the above for(for(loop)), that means the matrix is
      // already solved, so we directly return true from here; and print the solved
      // Sudoku.
      printSudoku(board);
      return true;
    }

    void printSudoku(vector<vector<char>> &board) {
      for (int r = 0; r < nrows; r++) {
         for (int c = 0; c < ncols; c++) {
           std::cout << board[r][c] << ",";
         }
         std::cout << std::endl;
      }
      std::cout << "======================\n";
    }
    
};

int main() {
  auto sol = Solution();
  vector<vector<char>> board =
    {{'5','3','.','.','7','.','.','.','.'},
     {'6','.','.','1','9','5','.','.','.'},
     {'.','9','8','.','.','.','.','6','.'},
     {'8','.','.','.','6','.','.','.','3'},
     {'4','.','.','8','.','3','.','.','1'},
     {'7','.','.','.','2','.','.','.','6'},
     {'.','6','.','.','.','.','2','8','.'},
     {'.','.','.','4','1','9','.','.','5'},
     {'.','.','.','.','8','.','.','7','9'}};
  sol.solveSudoku(board);
  return 0;
}
