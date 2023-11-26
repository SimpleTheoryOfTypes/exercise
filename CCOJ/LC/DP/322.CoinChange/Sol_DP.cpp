#include <vector>
#include <algorithm> 
using namespace std;

class Solution {
public:
    int coinChange(const vector<int>& coins, int amount) {

        if (amount <= 0)
            return 0;
        
        
      // dp[i] is the fewest number of coins to make up the number i;
      vector<int> dp(amount + 1, -1);
      for (const auto &c : coins) {
        if (c < dp.size())
          dp[c] = 1;
      }

      int minDenom = *min_element(coins.begin(), coins.end());
        
      if (amount < minDenom)
        return -1;
        
      for (int i = 0; i <= amount; i++) {
        if (dp[i] != -1)
          continue;

        if (i < minDenom)
          continue;

        int fewestNumber = INT_MAX;
        int found = false;//find a way to make up the amount
        for (const auto &c : coins) {
          if (i - c >= minDenom && dp[i - c] != -1) {
            fewestNumber = min(fewestNumber, dp[i - c] + 1);
            found = true;
          }
        }

        if (!found) {
          dp[i] = -1;
        } else {
          dp[i] = fewestNumber;   
        }


      }

      return dp[amount];
    }
};

int main() {
  auto sol = Solution();
  /* auto ans = sol.coinChange({1,2,5}, 11); */
  auto ans = sol.coinChange({2}, 3);
  return ans;
}

