#include <vector>
#include <algorithm>
#include <iostream>
using namespace std;

class Solution {
    vector<int> possibleProfits;
  public:
    int maxProfit(const vector<int>& prices) {
      int profit = 0;
      int prevState = 0;
      dfs(prevState, profit, 0, prices);
      printVector(possibleProfits);
      return *max_element(possibleProfits.begin(), possibleProfits.end());
    }

    void printVector(const vector<int>& V) {
      std::cout << std::endl;
      for (auto v : V)
        std::cout << v << ",";
      std::cout << std::endl;
    }

    // prevState:
    //   0: initial state (can buy or cool down).
    //   1: buy state (next state is sell or cool down).
    //   2: sell state (next state: must cool down)
    //   3: sell state (already had a cool down, next state: can be buy or cool down).
    // index: on the current day.
    // Given a previous state, at the current day pointing by the index, we need to
    // traverse down the search tree to update profit.
    void dfs(int &prevState, int &profit, int index, const vector<int>& prices) {
      if (index >= prices.size()) {
        possibleProfits.push_back(profit);
        return;
      }

      if (prevState == 0) {
        profit -= prices[index]; 
        prevState = 1; // BUY
        dfs(prevState, profit, index + 1, prices);
        prevState = 0;
        profit += prices[index];

        dfs(prevState, profit, index + 1, prices);
      } else if (prevState == 1) {
        dfs(prevState, profit, index + 1, prices);

        profit += prices[index];
        prevState = 2; // SELL
        dfs(prevState, profit, index + 1, prices);
        prevState = 1;
        profit -= prices[index];
      } else if (prevState == 2) {
        prevState = 3;//cool down, so skip at index.
        dfs(prevState, profit, index + 1, prices);
      } else if (prevState == 3) {
        profit -= prices[index];
        prevState = 1; // BUY
        dfs(prevState, profit, index + 1, prices);// must have a cool down after sell, so skip index to index + 1.
        prevState = 3;
        profit += prices[index];

        dfs(prevState, profit, index + 1, prices);
      } else {
        assert(false && "Unknown prevState.");
      }
    }
};

int main() {
  auto sol = Solution();
  auto ans = sol.maxProfit({1,2,3,0,2});
  return ans;
}
