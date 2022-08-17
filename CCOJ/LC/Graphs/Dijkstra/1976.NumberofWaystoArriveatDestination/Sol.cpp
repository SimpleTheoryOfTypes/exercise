#include <vector>
using namespace std;

// Dijkstra: https://cp-algorithms.com/graph/dijkstra.html
class Solution {
public:
    #define ll long long
    #define MOD 1000000007
    int countPaths(int n, const vector<vector<int>>& roads) {
      vector<vector<pair<ll, ll>>> g = vector<vector<pair<ll, ll>>>(n , vector<pair<ll,ll>>());
      for (const auto &r : roads) {
        g[r[0]].push_back({r[1], r[2]});
        g[r[1]].push_back({r[0], r[2]});
      }

      vector<ll> d = vector<ll>(n, LONG_MAX);// distance from node 0.
      d[0] = 0;
      vector<int> p = vector<int>(n, -1);//parent
      vector<bool> u = vector<bool>(n, false);//mark
      vector<int> nways = vector<int>(n, 0);
      nways[0] = 1;

      for (int i = 0; i < n; i++) {
        int v = -1;
        for (int j = 0; j < n; j++) {
          if (!u[j] && (v == -1 || d[j] < d[v])) {
            v = j;
          }
        }

        if (d[v] == LONG_MAX)
          break;

        u[v] = true;
        for (auto &e : g[v]) {
          ll to = e.first;
          ll weight = e.second;
          if (d[v] + weight < d[to]) {
            d[to] = d[v] + weight;
            p[to] = v;
            nways[to] = nways[v];
          } else if (d[v] + weight == d[to]){
            nways[to] = (nways[to] + nways[v]) % MOD;
          }
        }
      }

      return nways[n-1];
    }
};

int main() {
  auto sol = Solution();
  auto ans = sol.countPaths(2, {{1,0,10}});
  return ans;
}
