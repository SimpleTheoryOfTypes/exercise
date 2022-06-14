#include <vector>
#include <iostream>
#include <map>
using namespace std;
class Solution {
public:
    vector<vector<int>> permute(const vector<int>& nums) {
      vector<vector<int>> results = {};
      vector<int> ans = {};
      map<int, int> visited = {};
      generate(nums, visited, ans, results);
      return results; 
    }
    
    void generate(const vector<int>& nums, map<int, int>& visited, vector<int> &ans, vector<vector<int>>& results) {
      if (ans.size() == nums.size()) {
        results.push_back(ans);
        return;
      }

      for (auto n : nums) {
        if (visited.find(n) != visited.end())
          continue;

        ans.push_back(n);
        visited[n] = 1;
        generate(nums, visited, ans, results);
        visited.erase(visited.find(n));
        ans.erase(ans.end() - 1);
      }
    }
};


int main() {
  auto sol = Solution();
  auto results = sol.permute({1,2,3});
  for (auto r : results) {
    for (auto x : r)
      std::cout << " " << x;
    std::cout << std::endl;
  }

  return 0;
}
