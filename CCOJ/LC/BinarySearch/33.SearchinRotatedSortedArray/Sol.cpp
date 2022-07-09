#include <vector>
using namespace std;

class Solution {
  public:
    int search(const vector<int>& nums, int target) {
      if (nums.size() == 1) {
        if (nums[0] == target)
          return 0;
        return -1;
      }

      int lo = -1;
      int hi = -1;

      if (nums[0] < nums[nums.size() - 1]) {
        // The firt element less than the last element.
        // Not pivoting in the input array.
        lo = 0;
        hi = nums.size() - 1;
      } else {
        // The input array has pivoting. Identity the pivot
        // index, and decide which portion of the array we
        // should search for the target.
        int maxGap = INT_MIN;
        int pivotIndex = 0;
        for (int i = 0; i < nums.size() - 1; i++) {
          if (maxGap < nums[i] - nums[i + 1]) {
            maxGap = nums[i] - nums[i + 1];
            pivotIndex = i;
          }
        } 

        if (target >= nums[pivotIndex + 1] && target <= nums[nums.size() - 1]) {
          // binary search the right portion.
          lo = pivotIndex + 1;
          hi = nums.size() - 1;
        } else {
          // binary search the left portion.
          lo = 0;
          hi = pivotIndex;
        }
      }

      while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (nums[mid] > target)
          hi = mid - 1;
        else if (nums[mid] < target)
          lo = mid + 1;
        else
          return mid;
      }

      return -1;
    }
};

int main() {
  auto sol = Solution();
  /* auto ans = sol.search({4,5,6,7,0,1,2}, 0); */
  auto ans = sol.search({1,3}, 3);
  return ans;
}
