#include <string>
#include <map>
using namespace std;

class Solution {
    map<tuple<int, int, int>, bool> dp;
public:
    bool isInterleave(string s1, string s2, string s3) {
      if (s3.size() != s1.size() + s2.size())
        return false;

      return dfs(s1, s2, s3, 0, 0, 0);
    }

    bool dfs(string &s1, string &s2, string &s3, int i, int j, int k) {
      if (k >= s3.size())
        return true;

      if (s3[k] == s1[i]) {
        bool flag = false;
        const auto tp = make_tuple(i + 1, j, k + 1);
        if (dp.find(tp) != dp.end()) {
          flag = dp[tp];
        } else {
          flag = dfs(s1, s2, s3, i + 1, j, k + 1);
          dp[tp] = flag;
        }
        if (flag)
          return true;
      }

      if (s3[k] == s2[j]) {
        bool flag = false;
        const auto tp = make_tuple(i, j + 1, k + 1);
        if (dp.find(tp) != dp.end())
          flag = dp[tp];
        else {
          flag = dfs(s1, s2, s3, i, j + 1, k + 1);
          dp[tp] = flag;
        }
        if (flag)
          return true;
      }
      
      return false;
    }
};

int main() {
  auto sol = Solution();
  auto ans = sol.isInterleave("bcc", "ca", "baccc");
  return ans;
}
