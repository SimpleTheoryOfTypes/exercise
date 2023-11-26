//https://leetcode.com/contest/weekly-contest-305/problems/check-if-there-is-a-valid-partition-for-the-array/
//2369. Check if There is a Valid Partition For The Array
#include <vector>
using namespace std;

class Solution {
    vector<int> dp;
public:
    bool validPartition(const vector<int>& nums) {
      dp = vector<int>(nums.size() + 1, -1);
      return dfs(nums, 0);
    }

    int dfs(const vector<int> &nums, int index) {
      if (index == nums.size())
        return true;

      if (dp[index] != -1)
        return dp[index];

      if (index + 1 < nums.size() && nums[index] == nums[index+1]) {
        int flag = dfs(nums, index + 2);
        dp[index + 2] = flag;
        if (flag) return flag;
      }

      if (index + 2 < nums.size() && nums[index] == nums[index+1] && nums[index+1] == nums[index+2]) {
        int flag = dfs(nums, index + 3);
        dp[index + 3] = flag;
        if (flag) return flag;
      }

      if (index + 2 < nums.size() && nums[index] + 1 == nums[index+1] && nums[index+1] + 1 == nums[index+2]) {
        int flag = dfs(nums, index + 3);
        dp[index + 3] = flag;
        if (flag) return flag;
      }
      return false;
    }
};

int main() {
  auto sol = Solution();
  auto ans = sol.validPartition({803201,803201,803201,803201,803202,803203});
  return ans;
}
