#include <vector>
#include <queue>
#include <map>
#include <deque>
using namespace std;

class Solution {
    map<int, int> color;
    map<int, int> discover;
public:
    int numOfMinutes(int n, int headID, const vector<int>& manager, const vector<int>& informTime) {
      vector<vector<int>> g = vector<vector<int>>(n+1, vector<int>());
      for (int i = 0; i < manager.size(); i++) {
        if (manager[i] != -1) {
          g[manager[i]].push_back(i);
          color[i] = 0;
          discover[i] = INT_MAX;
        }
      }

      queue<int> Q;
      Q.push(headID);
      discover[headID] = 0;
      while (!Q.empty()) {
        const auto u = Q.front();
        Q.pop();

        for (const auto &nbr : g[u]) {
          if (color[nbr] == 0) {
            Q.push(nbr);
            discover[nbr] = discover[u] + 1;
            color[nbr] = 1;
          }
        }
      }

      int ans = 0;
      for (int level = 0; level < n; level++) {
        int maxInformTime = -1;
        for (const auto &[n, d] : discover) {
          if (d == level && informTime[n] > maxInformTime)
            maxInformTime = informTime[n];
        }
        if (maxInformTime != -1) 
          ans += maxInformTime;
      }
      return ans;
    }
};

int main() {
  auto sol = Solution();
  auto ans = sol.numOfMinutes(6, 2, {2,2,-1,2,2,2}, {0,0,1,0,0,0});
  return ans;
}
