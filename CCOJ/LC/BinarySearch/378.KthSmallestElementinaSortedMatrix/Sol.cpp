#include <vector>
#include <iostream>
using namespace std;

class Solution {
public:
    int kthSmallest(const vector<vector<int>>& matrix, int k) {
      int nrows = matrix.size();
      int ncols = matrix[0].size();
      int leftVal = matrix[0][0], rightVal = matrix[nrows - 1][ncols - 1];
      // (invariant) the kth Smallest value must be in the close interval of [left, right]
      while (leftVal + 1 < rightVal) {
        int mid = (leftVal + rightVal) / 2;
        if (good(mid, matrix) <= k - 1) {
          // (invariant) must be in [mid+1, right]
          leftVal = mid;
          // (invariant) must be in [left, right]
        } else {
          // (invariant) musth be in [left, mid] since count of mid >= k 
          rightVal = mid;
          // (invariant) must be in [left, right]
        }
        // (invariant) must be in [left, right]
      }
      // (invariant) the kth smallest value must be in [left, right] && left >= right.
      // Therefore, left == right == the kth smallest value.
      /* assert(leftVal == rightVal); */
      int x = good(leftVal, matrix);
      int y = good(rightVal, matrix);

      std::cout << "(SimpleTheoryOfTypes) ans is either " << leftVal << " or " << rightVal << std::endl;
      return leftVal;
    }

    bool exists(const vector<vector<int>> &matrix, int val) {
      for (const auto &v : matrix) {
        for (const auto &x : v) {
          if (x == val)
            return true;
        }
      }
      return false;
    }

    int good(int mid, const vector<vector<int>>& matrix) {
      int count = 0;
      for (const auto &v : matrix) {
        count += bisect_left(v, mid);
      }
      return count;
    }

    int bisect_left(const vector<int> &A, int target) {
      // simple case when target is bigger than every element in A.
      if (target > A[A.size() - 1])
        return A.size();

      // Find the index to insert target in the sorted array A such that A
      // is still sorted after the insertion. Same as python's bisect_left.
      int left = 0, right = A.size() - 1;
      // (invariant): insertion point must be in the close interval: [left, right]
      while (right > left + 1) {
        int mid = (left + right) / 2;
        if (A[mid] >= target) {
          // (invariant) must be in [left, mid - 1]
          right = mid - 1;
          // (invariant) must be in [left, right] 
        } else if (A[mid] < target) {
          // (invariant) must be in [mid, right] 
          left = mid; 
          // (invariant) must be in [left, right] 
        }
        // (invariant) must be in [left, right]
      }
      // [invariant] mustbe in [left, right] && left + 1 >= right
      // Therefore, the insertion index must be also in [left, left + 1], since [left, right] is a subset of [left, left + 1].
      // So, we've narrowed down the insertion index to either left or left + 1:
      if (target <= A[left])
        return left;
      return left + 1;
    }
};

int main() {
  auto sol = Solution();
  /* auto ans = sol.bisect_left({12,13,15}, 12); */
  /* auto ans = sol.kthSmallest({{1,5,9},{10,11,13},{12,13,15}}, 8); */
  /* auto ans = sol.kthSmallest({{-5}}, 1); */
  auto ans = sol.kthSmallest({{1,2},{1,3}}, 2);
  return ans;
}
