#include <vector>
#include <map>
#include <string>
#include <iostream>
using namespace std;

class Solution {
  vector<int> directions = {-1, 1};
  bool Found = false;
  int nrows;
  int ncols;
public:
    void Constructor(const vector<vector<char>>& board) {
      nrows = board.size();
      ncols = board[0].size();
    }

    bool exist(const vector<vector<char>>& board, const string &word) {
      Constructor(board);

      // Initialize bitMask to a nrows x ncols grid of all 0's.
      // Create a vector containing "nrows" vectors each of size "ncols".
      vector<vector<int>> bitMask(nrows, vector<int>(ncols));
      for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
          bitMask[i][j] = 0;
        }
      }

      /* print2DVector(bitMask); */
      string partialResult;

      for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
          partialResult = board[i][j];
          reset2DVector(bitMask);
          bitMask[i][j] = 1;
          //print2DVector(bitMask);
          //std::cout << "(SimpleTheoryOfTypes) Staring from " << i << ", " << j << ": " << partialResult << std::endl;
          generate(word, board, partialResult, bitMask, i, j);
        }
      }

      return Found;
    }

    void print2DVector(const vector<vector<int>> &x) {
      for (auto v : x) {
        for (auto r : v) {
          std::cout << "" << r << ",";
        }
        std::cout << "\n";
      }
    }

    void reset2DVector(vector<vector<int>> &x) {
      for (auto &v : x) {
        for (auto &r : v) {
          r = 0;
        }
      }
    }

    void generate(const string &word, const vector<vector<char>>& board, string &partialResult, vector<vector<int>> &bitMask, int posI, int posJ) {
      if (partialResult[partialResult.size()-1] != word[partialResult.size()-1])
        return;

      if (partialResult == word) {
        Found = true;
      }

      // posI and posJ are the current positions.
      // bitMask encodes cells that have been traversed (0: not, 1: traversed).
      // partialResult collects partial string collected so far.
      // Move horizontally:
      for (auto i : directions) {
        auto nextI = posI + i;
        if (nextI < nrows && nextI >= 0) {
          if (bitMask[nextI][posJ] == 0) {
            partialResult.insert(partialResult.end(), board[nextI][posJ]);
            bitMask[nextI][posJ] = 1;
            generate(word, board, partialResult, bitMask, nextI, posJ);
            partialResult.erase(partialResult.end() - 1);
            bitMask[nextI][posJ] = 0;
          }
        }
      }

      // Move vertically:
      for (auto j : directions) {
        auto nextJ = posJ + j;
        if (nextJ < ncols && nextJ >= 0) {
          if (bitMask[posI][nextJ] == 0) {
            partialResult.insert(partialResult.end(), board[posI][nextJ]);
            bitMask[posI][nextJ] = 1;
            generate(word, board, partialResult, bitMask, posI, nextJ);
            partialResult.erase(partialResult.end() - 1);
            bitMask[posI][nextJ] = 0;
          }
        }
      }
    }
};

int main() {
  auto sol = Solution();
  /* auto ans = sol.exist({{'A','B','C','E'},{'S','F','C','S'},{'A','D','E','E'}}, "ABCCED"); */
  /* auto ans = sol.exist({{'A'}}, "A"); */
  //{{"L","L","A","B","L","D"},{"G","A","B","A","L","L"},{"A","B","C","B","J","A"},{"L","E","D","H","I","L"}} "GABALJIHDCBA"
  auto ans = sol.exist({{'L','L','A','B','L','D'},{'G','A','B','A','L','L'},{'A','B','C','B','J','A'},{'L','E','D','H','I','L'}}, "GABALJIHDCBA");

  return ans;
}
