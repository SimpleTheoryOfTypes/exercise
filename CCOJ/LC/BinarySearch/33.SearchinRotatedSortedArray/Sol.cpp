#include <vector>
using namespace std;

class Solution {
  public:
    int search(const vector<int>& nums, int target) {
      int lo = 0;
      int hi = nums.size() - 1;

      while (lo <= hi) {
        int mid = (lo + hi) / 2;

        if (target == nums[mid])
          return mid;

        // Right portion
        if (nums[mid] < nums[hi]) {
          if (target < nums[mid] || target > nums[hi]) {
            hi = mid - 1;
          } else {
            lo = mid + 1;
          }
        } else {
          if (target > nums[mid] || target <= nums[hi]) {
            lo = mid + 1;
          } else {
            hi = mid - 1;
          }
        }
      }
      return -1;
    }
};

int main() {
  auto sol = Solution();
  /* auto ans = sol.search({4,5,6,7,0,1,2}, 0); */
  auto ans = sol.search({3,1,2}, 3);
  return ans;
}
