#include <vector>
#include <string>
#include <iostream>
using namespace std;
class Solution {
    vector<int> dp;
    vector<bool> dpFlags;// true if dp[i] is set, else false
    int N = 0;
public:
    int numDecodings(string s) {
      if (s.empty() || s[0] == '0')
        return 0;

      N = s.size();
      dp = vector<int>(N, 0);
      dpFlags = vector<bool>(N, false);
      dfs(s, 0);
      printVector(dp);
      return dp[0]; 
    }

    int dfs(string &s, int i) {
      if (dpFlags[i] == true)
        return dp[i];

      if (s[i] == '0') {
        dp[i] = 0;
        return dp[i];
      }

      if (i + 1 < N) {
        if (s[i + 1] != '0')
          dp[i] = dfs(s, i + 1);

        if ((s[i] - '0') * 10 + (s[i + 1] - '0') <= 26) {
          if (i + 2 < N)
            dp[i] += dfs(s, i + 2);
          else
            dp[i] += 1;
        }
      } else {
        dp[i] = 1;
      }

      dpFlags[i] = true;
      return dp[i];
    }

    void printVector(const vector<int> &V) {
      for (const auto x : V)
        std::cout << x << ",";
      std::cout << std::endl;
    }
};

int main() {
  auto sol = Solution();
  auto ans = sol.numDecodings("2611055971756562");
  return ans;
}
