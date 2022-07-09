#include <vector>
using namespace std;

class Solution {
  public:
    long long minEatingSpeed(vector<int>& piles, long long h) {
      long long lo = 1;
      long long hi = *max_element(piles.begin(), piles.end());

      while (lo <= hi) {
        long long K = (lo + hi) / 2;
        long long totalHours = computeHours(piles, K);
        if (totalHours <= h) {
          // Need to reduce speed, i.e., K too big.
          hi = K - 1;
        } else if (totalHours > h) {
          // totalHours too large, increase eating speed K.
          lo = K + 1;
        }
      }

      return lo;
    }

    long long computeHours(vector<int> & piles, long long K) {
      long long totalHours = 0;
      for (const auto &pile : piles) {
        if (pile % K == 0)
          totalHours += pile / K;
        else
          totalHours += pile / K + 1;
      }
      return totalHours;
    }
};

int main() {
  auto sol = Solution();
  /* auto ans = sol.minEatingSpeed({3,6,7,11}, 8); */
  auto ans = sol.minEatingSpeed({805306368,805306368,805306368}, 1000000000);
  return ans;
}
