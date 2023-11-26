#include <vector>
#include <iostream>
using namespace std;

class Solution {
public:
    uint64_t peopleAwareOfSecret(int n, int delay, int forget) {
      vector<uint64_t> K(n, 0);//K[i] represents the number of people who know the secret the first time on the ith day.

      // Set up initial condition.
      K[0] = 1;

      // Iterate till the nth day.
      for (int i = delay; i < n; i++) {
        for (int x = delay; x < forget; x++) {
          if (i - x >= 0) {
            // On the ith day, everyone who had knew the secret that post-delay,
            // pre-forget can tell the secret to a new person each!
            K[i] += K[i - x] % (1000000000 + 7);
          }
        }
        K[i] %= (1000000000 + 7);
      }

      uint64_t ans = 0;
      for (int x = 0; x < forget; x++) {
        ans += K[(n - 1) - x];
      }

      return ans % (1000000000 + 7);
    }
};

int main() {
  auto sol = Solution();
  /* auto ans = sol.peopleAwareOfSecret(6,2,4); */
  auto ans = sol.peopleAwareOfSecret(684, 18, 496);
  return ans;
}
