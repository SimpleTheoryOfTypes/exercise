#include <vector>
#include <numeric>
#include <iostream>
#include <algorithm>

using namespace std;

class Solution {
public:
    vector<vector<int>> combinationSum(const vector<int>& candidates, int target) {
      vector<vector<int>> results = {};
      vector<int> ans = {};
      int partialSum = 0;
      generate(candidates, target, ans, partialSum, results);
      return results;
    }

    void generate(const vector<int>& candidates, int target, vector<int> &ans, int partialSum, vector<vector<int>>& results) {
       if (partialSum > target)
         return;

       if (partialSum == target) {
         vector<int> ansCopy = ans;
         std::sort(ansCopy.begin(), ansCopy.end());
         if (!answer_already_exists(ansCopy, results)) {
           results.insert(results.end(), ansCopy);
           if (results.size() > 150)
             assert(0);
         }
         return;
       }

       for (auto c : candidates) {
         ans.insert(ans.end(), c);
         partialSum += c;
         generate(candidates, target, ans, partialSum, results);
         ans.erase(ans.end() - 1);
         partialSum -= c;
       } 
    }

    void printVector(vector<int>& X) {
      std::cout << "(";
      for (auto x : X) {
        std::cout << x << ",";
      }
      std::cout << ")" << std::endl;
    }

    bool answer_already_exists(const vector<int>& ans, vector<vector<int>>& results) {
      for (auto x : results) {
        if (same_answer(ans, x))
          return true;
      }
      return false;
    }

    bool same_answer(const vector<int>& ansX, const vector<int>& ansY) {
      if (ansX.size() != ansY.size())
        return false;

      // Assume ansX and ansY are sorted.
      for (size_t i = 0; i < ansX.size(); i++) {
        auto x = ansX[i];
        auto y = ansY[i];
        if (x != y) {
          return false;
        }
      }

      return true; 
    }
};

int main() {
  auto sol = Solution();
  /* auto ans = sol.combinationSum({2,3,6,7}, 7); */
  /* auto ans = sol.combinationSum({2,3,5}, 8); */
  /* auto ans = sol.combinationSum({1,2}, 4); */
  /* auto ans = sol.combinationSum({2,7,6,3,5,1}, 9); */
  auto ans = sol.combinationSum({100,200,4,12}, 400);

  std::cout << "(SimpleTheoryOfTypes) size = " << ans.size() << std::endl;
  for (auto X : ans) {
    std::cout << "(SimpleTheoryOfTypes)" << std::endl;
    for (auto x : X) {
      std::cout << " " << x << ", ";
    }
    std::cout << std::endl;
  }
  return 0;
}
