#include <vector>
#include <algorithm>
#include <iostream>
using namespace std;

class Solution {
public:
    vector<int> minInterval(vector<vector<int>>& intervals, vector<int>& queries) {
      // Sort intervals keyed at the starting point:
      std::sort(intervals.begin(), intervals.end(), [] (const auto &lhs, const auto &rhs) {
        return lhs[0] < rhs[0];
      });

      // Sort queries with indices by query value in ascending order.
      vector<pair<int, int>> qni;// query and its index.
      for (int i = 0; i < queries.size(); i++) {
        qni.push_back(make_pair(queries[i], i));
      }
      std::sort(qni.begin(), qni.end(), [] (const auto &lhs, const auto &rhs) {
        return lhs.first < rhs.first;
      });

      vector<pair<int, int>> pq; // pair = <interval len, interval endpoint>
      vector<pair<int, int>> results; // pair = <query index, min interval len>
      int idx = 0; // interval index
      for (int i = 0; i < qni.size(); i++) {
        const auto &[query, index] = qni[i];
        while (idx < intervals.size()) {
          const auto s = intervals[idx][0];
          const auto e = intervals[idx][1];
          if (query < s)
            break;

          pq.push_back(make_pair(e-s+1, e));
          push_heap(pq.begin(), pq.end(), [] (const auto &lhs, const auto &rhs) {
              return lhs.first > rhs.first;
          });
          idx += 1;
        }


        bool foundResult = false;
        while (!pq.empty()) {
          const auto [L, E] = pq[0]; 
          if (E < query) {
            pop_heap(pq.begin(), pq.end(), [] (const auto &lhs, const auto &rhs) {
                return lhs.first > rhs.first;
            });
            pq.pop_back();
          } else {
            results.push_back(make_pair(index, L));
            foundResult = true;
            break;
          }
        }
        if (!foundResult) {
          results.push_back(make_pair(index, -1));
        }
      }

      std::sort(results.begin(), results.end(), [] (const auto &lhs, const auto &rhs) {
          return lhs.first < rhs.first; // sort results by query index.
      });

      vector<int> ans;
      for (const auto &[qidx, len] : results) {
        ans.push_back(len);
      }
        
      return ans;
    }
};

int main() {
  auto sol = Solution();
  vector<vector<int>> intervals = {{1,4},{2,4},{3,6},{4,4}};
  vector<int> queries = {2,3,4,5}; 
  //vector<vector<int>> intervals = {{2,3},{2,5},{1,8},{20,25}};
  //vector<int> queries = {2,19,5,22}; 
  auto ans = sol.minInterval(intervals, queries);
  for (auto &a : ans)
    std::cout << a << ",";
  std::cout << "\n";
  return 0;
}
