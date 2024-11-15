#include <vector>
#include <iostream>
using namespace std;

// Mathematics:
// ===========
// A[mid] >= A[lo] && A[lo] > A[hi] => min on the right: [lo = mid + 1, hi]
// A[mid] >= A[lo] && A[lo] < A[hi] => min on the left : [lo, hi = mid - 1]
// A[mid] < A[lo]  => if A[mid - 1] < A[mid], then min on the left
//                                      else, return A[mid].

class Solution {
public:
    int findMin(const vector<int>& nums) {
      int lo = 0;
      int hi = nums.size() - 1;
      
      while (lo <= hi) {
        int mid = (lo + hi) / 2;
        
        if (nums[mid] >= nums[lo]) {
          if (nums[lo] > nums[hi])
            lo = mid + 1;
          else
            hi = mid - 1;
        } else {
          if ((mid - 1 >= 0) && nums[mid - 1] < nums[mid])
            hi = mid - 1;
          else
            return nums[mid];
        }
      }

      return nums[lo];
    }
};

int main() {
  auto sol = Solution();
  auto ans = sol.findMin({0,1,2,4,5,6,7});
  std::cout << "(SimpleTheoryOfTypes) ans = " << ans << std::endl;
  return ans;
}
