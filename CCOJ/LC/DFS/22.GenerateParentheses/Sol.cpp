#include <vector>
#include <string>
#include <iostream>
using namespace std;

// Brute Force:
// To generate all sequences, we use a recursion. All sequences of length n is just '('
// plus all sequences of length n-1, and then ')' plus all sequences of length n-1.
// To check whether a sequence is valid, we keep track of balance, the net number of
// opening brackets minus closing brackets. If it falls below zero at any time, or
// doesn't end in zero, the sequence is invalid - otherwise it is valid.
class Solution {
public:
    vector<string> generateParenthesis(int n) {
      vector<string> results = {};
      string initialStr = "";
      generate(initialStr, n, results);

      return results;
    }

    void generate(string &A, int n, vector<string> &results) {
      if (A.length() == 2*n) {
        if (is_valid(A)) {
          results.push_back(A);
        }
      } else {
        A.insert(A.end(), '(');
        generate(A, n, results);
        A.erase(A.end()-1);
        A.insert(A.end(), ')');
        generate(A, n, results);
        A.erase(A.end()-1);
      }
    }

    bool is_valid(string &A) {
      if (A[0] != '(')
        return false;

      int balance = 0;
      for (size_t i = 0; i < A.size(); i++) {
        if (balance == 0 && A[i] != '(')
          return false;

        if (A[i] == '(')
          balance += 1;
        else if (A[i] == ')')
          balance -= 1;

      }

      if (balance != 0)
        return false;

      return true;
    }

};

int main() {
  auto sol = Solution();
  int n = 3;
  vector<string> answer = sol.generateParenthesis(n);
  for (auto &ans : answer)
    std::cout << "(SimpleTheoryOfTypes) ans = " << ans << std::endl;

  return answer.size(); 
}
