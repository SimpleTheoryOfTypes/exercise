#include <vector>
#include <algorithm>
#include <iostream>
using namespace std;

class Solution {
    vector<int> possibleProfits;
  public:
    int maxProfit(const vector<int>& prices) {
      int profit = 0;
      int state = 0;
      dfs(state, profit, 0, prices);
      printVector(possibleProfits);
      return *max_element(possibleProfits.begin(), possibleProfits.end());
    }

    void printVector(const vector<int>& V) {
      std::cout << std::endl;
      for (auto v : V)
        std::cout << v << ",";
      std::cout << std::endl;
    }

    void dfs(int &state, int &profit, int index, const vector<int>& prices) {
      if (index >= prices.size()) {
        possibleProfits.push_back(profit);
        return;
      }

      if (state == 0) {
        profit -= prices[index]; 
        state = 1; // BUY
        dfs(state, profit, index + 1, prices);
        state = 0;
        profit += prices[index];

        dfs(state, profit, index + 1, prices);
      } else if (state == 1) {
        dfs(state, profit, index + 1, prices);

        profit += prices[index];
        state = 2; // SELL
        dfs(state, profit, index + 1, prices);
        state = 1;
        profit -= prices[index];
      } else if (state == 2) {
        state = 3;//cool down, so skip at index.
        dfs(state, profit, index + 1, prices);
      } else if (state == 3) {
        profit -= prices[index];
        state = 1; // BUY
        dfs(state, profit, index + 1, prices);// must have a cool down after sell, so skip index to index + 1.
        state = 3;
        profit += prices[index];

        dfs(state, profit, index + 1, prices);
      } else {
        assert(false && "Unknown state.");
      }
    }
};

int main() {
  auto sol = Solution();
  auto ans = sol.maxProfit({1,2,3,0,2});
  return ans;
}
