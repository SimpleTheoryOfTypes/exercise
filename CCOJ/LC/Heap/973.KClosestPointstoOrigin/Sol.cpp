#include <vector>
#include <queue>
using namespace std;

class Solution {
public:
    vector<vector<int>> kClosest(const vector<vector<int>>& points, int k) {
      auto comp = [] (const vector<int> &x, const vector<int> &y) {
        int distx = x[0] * x[0] + x[1] * x[1];
        int disty = y[0] * y[0] + y[1] * y[1];
        return distx > disty;
      };
      priority_queue<vector<int>, vector<vector<int>>, decltype(comp)> pq(comp);
     
      for (auto &p : points) {
        pq.push(p);
      }

      int count = 0;
      vector<vector<int>> ans;
      while (!pq.empty()) {
        auto u = pq.top();
        ans.push_back(u);
        pq.pop();
        count += 1;
        if (count == k)
          break;
      }

      return ans;
    }
};

int main() {
  auto sol = Solution();
  auto ans = sol.kClosest({{3,3},{5,-1},{-2,4}}, 2);
  return 0;
}
