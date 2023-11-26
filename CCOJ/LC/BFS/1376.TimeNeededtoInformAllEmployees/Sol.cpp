#include <vector>
#include <queue>
#include <map>
#include <deque>
#include <set>
using namespace std;

class Solution {
    map<int, int> color;
    map<int, int> discover;
public:
    int numOfMinutes(int n, int headID, const vector<int>& manager, const vector<int>& informTime) {
      vector<vector<pair<int, int>>> g = vector<vector<pair<int,int>>>(n, vector<pair<int,int>>());
      for (int i = 0; i < manager.size(); i++) {
        if (manager[i] != -1) {
          g[manager[i]].push_back({i, informTime[manager[i]]});
        }
      }

      vector<int> d = vector<int>(n, INT_MAX);
      vector<int> p = vector<int>(n, -1);
      d[headID] = 0;
      set<pair<int, int>> q;
      q.insert({0, headID});
      while (!q.empty()) {
        int v = q.begin()->second;
        q.erase(q.begin());

        for (const auto &[to, wt] : g[v]) {
          if (d[v] + wt < d[to]) {
            q.erase({d[to], to});
            d[to] = d[v] + wt;
            p[to] = v;
            q.insert({d[to], to});
          }
        }
      }
      return *max_element(d.begin(), d.end());
    }
};

int main() {
  auto sol = Solution();
  auto ans = sol.numOfMinutes(6, 2, {2,2,-1,2,2,2}, {0,0,1,0,0,0});
  return ans;
}
