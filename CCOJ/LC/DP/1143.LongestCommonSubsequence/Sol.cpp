#include <string>
#include <vector>
#include <iostream>
using namespace std;

class Solution {
  int dp[1000][1000];
  public:
    int longestCommonSubsequence(const string &text1, const string &text2) {
      int m = text1.size();
      int n = text2.size();

      memset(dp, -1, sizeof(dp));
      return lcs(text1, text2, m-1, n-1);
    }

    int lcs(const string &text1, const string &text2, int i, int j) {
      if (i == -1 || j == -1)
        return 0;

      if (dp[i][j] != -1)
        return dp[i][j];

      if (text1[i] == text2[j]) {
        dp[i][j] = lcs(text1, text2, i-1, j-1) + 1;
        return dp[i][j];
      }

      dp[i][j] = max(lcs(text1, text2, i-1, j), lcs(text1, text2, i, j-1));
      return dp[i][j];
    }
};

int main() {
  auto sol = Solution();
  auto ans = sol.longestCommonSubsequence("abcde", "ace");
  return ans;
}
