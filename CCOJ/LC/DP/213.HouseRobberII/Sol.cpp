#include <vector>
using namespace std;

class Solution {
  public:
    int rob(const vector<int>& nums) {
      if (nums.empty())
        return 0;
      if (nums.size() == 1)
        return nums[0];

      if (nums.size() == 2)
        return max(nums[0], nums[1]);

      if (nums.size() == 3)
        return max(nums[0], max(nums[1], nums[2]));

      vector<int> subvector0(nums.begin(), nums.end() - 1);
      int profit0 = robHelper(subvector0);
      vector<int> subvector1(nums.begin() + 1, nums.end());
      int profit1 = robHelper(subvector1);
      return max(profit0, profit1);
    }

    // This is the solution for House Robber I (the acyclic case).
    // We just run it twice on nums, once on nums[0:end-1], once
    // on nums[1:end], b/c nums[0] and nums[end-1] are connected
    // cyclicly, we are not allowed to rob both of them at the
    // same time.
    int robHelper(vector<int>& nums) {
      assert (nums.size() >= 3);
      vector<int> dp(nums.size(), 0);
      dp[0] = nums[0];
      dp[1] = max(nums[0], nums[1]);
      int maxProfit = 0xFFFFFFFF;
      for (int i = 2; i < nums.size(); i++) {
        dp[i] = max(dp[i-2] + nums[i], dp[i-1]);
        maxProfit = max(dp[i], maxProfit);
      }
      return maxProfit;
    }
};

int main() {
  auto sol = Solution();
  /* auto ans = sol.rob({1,2,3,1}); */
  auto ans = sol.rob({2,3,2});
  return ans;
}
