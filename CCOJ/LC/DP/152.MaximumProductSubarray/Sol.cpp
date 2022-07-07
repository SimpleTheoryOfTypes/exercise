#include <vector>
#include <numeric>
#include <iostream>
using namespace std;

class Solution {
  public:
    int maxProduct(const vector<int>& nums) {
      if (nums.empty())
        return 0;

      if (nums.size() == 1)
        return nums[0];

      vector<int> dpMax(nums.size(), INT_MIN);
      vector<int> dpMin(nums.size(), INT_MAX);

      dpMax[0] = nums[0];
      dpMin[0] = nums[0];

      int ans = dpMax[0];
      for (int i = 1; i < nums.size(); i++) {
        dpMax[i] = max(max(dpMin[i-1] * nums[i], nums[i]), dpMax[i-1] * nums[i]);
        dpMin[i] = min(min(dpMin[i-1] * nums[i], nums[i]), dpMax[i-1] * nums[i]);

        if (dpMax[i] > ans)
          ans = dpMax[i];
      }

      for (int i = 0; i < nums.size(); i++)
        std::cout << dpMax[i] << ",";
      std::cout << "\n";
      for (int i = 0; i < nums.size(); i++)
        std::cout << dpMin[i] << ",";

      return ans;
    }
};

int main() {
  auto sol = Solution();
  /* auto ans = sol.maxProduct({2,3,-2,4}); */
  /* auto ans = sol.maxProduct({-2,0,-1}); */
  /* auto ans = sol.maxProduct({-1,-2,-9,-6}); */
  auto ans = sol.maxProduct({2,-1,1,1});
  return ans;
}

