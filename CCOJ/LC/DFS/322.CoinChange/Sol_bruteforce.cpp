#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>
using namespace std;

class Solution {
public:
    int coinChange(const vector<uint64_t>& coins, int amount) {
      vector<uint64_t> choices {};
      auto minimumLen = UINT_MAX;
      backtrack(choices, coins, amount, minimumLen);
      return minimumLen == UINT_MAX ? -1 : minimumLen;
    }

    void backtrack(vector<uint64_t> &choices, const vector<uint64_t>& coins, const int amount, unsigned &currLen) {
      if (std::accumulate(choices.begin(), choices.end(), 0) > amount) {
        return;
      }

      if (choices.size() >= currLen) {
        return;
      }

      if (std::accumulate(choices.begin(), choices.end(), 0) == amount) {
        currLen = std::min<unsigned>(choices.size(), currLen);
        printVector(choices);
        return;
      }

      for (auto c : coins) {
        if (c > amount)
          continue;
        choices.push_back(c);
        backtrack(choices, coins, amount, currLen);
        choices.erase(choices.end() - 1);
      }
    }

    void printVector(vector<uint64_t> &v) {
      for (auto x : v)
        std::cout << "" << x << ",";
      std::cout << "\n";
    }
};

int main() {
  auto sol = Solution();
  /* auto ans = sol.coinChange({1,2,5}, 11); */
  /* auto ans = sol.coinChange({2}, 3); */
  /* auto ans = sol.coinChange({1}, 0); */
  /* auto ans = sol.coinChange({2147483647,1}, 2); */
  auto ans = sol.coinChange({1,2,5}, 100);
  return ans;
}
