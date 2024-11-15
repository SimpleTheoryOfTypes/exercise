#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;

class Solution {
  public:
    int lengthOfLIS(const vector<int>& nums) {
      int N = nums.size();
      // dp[i] represents the LIS ending with i;
      vector<int> dp(N, 1); 

      // dp[i] = max of:
      //   1. dp[j] + 1 for all j < i and nums[j] < nums[i]
      //   2. 1
      for (size_t i = 1; i < N; i++) {
        int maxL = 1;
        for (size_t j = 0; j < i; j++) {
          if (nums[j] < nums[i]) {
            maxL = max(maxL, dp[j] + 1);
          }
        }
        dp[i] = maxL;
      }

      return *max_element(dp.begin(), dp.end());
    }
};

int main() {
  auto sol = Solution();
  /* auto ans = sol.lengthOfLIS({10,9,2,5,3,7,101,18}); */
  auto ans = sol.lengthOfLIS({1,3,6,7,9,4,10,5,6});
  return ans;
}
