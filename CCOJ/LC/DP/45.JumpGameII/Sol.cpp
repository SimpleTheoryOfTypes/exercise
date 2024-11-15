#include <vector>
#include <algorithm>
using namespace std;

// Let f(i) denote the steps it takes to reach the last index from i.
// f(i) = Min[f(i+s) + 1], for s from 1 to nums[i].
class Solution {
public:
    int jump(const vector<int>& nums) {
      int N = nums.size();
      vector<int> ans(N, 0);
      for (int i = N - 1; i >= 0; i--) {
        if (i == N -1) {
          ans[i] = 0;
          continue;
        }

        if (nums[i] <= 0) {
          ans[i] = INT_MAX;
          continue;
        }

        int myMinSteps = N;
        for (int s = 1; s <= nums[i]; s++) {
          if (i + s < N)
            myMinSteps = min(ans[i + s], myMinSteps);
        }
        ans[i] = myMinSteps + 1;
      }

      return ans[0];
    }
};

int main() {
  auto sol = Solution();
  /* auto ans = sol.jump({2,3,1,1,4}); */
  auto ans = sol.jump({2,1});
  return ans;
}

