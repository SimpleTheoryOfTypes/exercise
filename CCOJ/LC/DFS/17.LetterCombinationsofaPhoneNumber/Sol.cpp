#include <vector>
#include <string>
#include <map>
#include <iostream>
using namespace std;

// Backtracking approach:
// 1. Initial conditions.
// 2. Terminating conditions.
// 3. Set up recursion.
class Solution {
  map<char, string> TheMap = {
    {'2', "abc"},
    {'3', "def"},
    {'4', "ghi"},
    {'5', "jkl"},
    {'6', "mno"},
    {'7', "pqrs"},
    {'8', "tuv"},
    {'9', "wxyz"}
  };
public:
    vector<string> letterCombinations(string digits) {
      vector<string> results = {};
      string initStr = ""; // initial condition.
      generate(initStr, digits, 0, results);
      return results;
    }

    void generate(string &A, const string &digits, int index, vector<string>& results) {
      if (digits == "")
        return;

      if (index == digits.length()) { // Terminating condition.
        results.push_back(A);
        return;
      }

      auto mappedStr = TheMap[digits[index]]; 
      for (size_t i = 0; i < mappedStr.length(); i++) {
        A.insert(A.end(), mappedStr[i]);
        generate(A, digits, index + 1, results);
        A.erase(A.end()-1);
      }
    }
};

int main() {
  auto sol = Solution();
  auto X = sol.letterCombinations("23");
  for (auto x : X)
    std::cout << "(SimpleTheoryOfTypes) x = " << x << std::endl;

  return 0;
}
