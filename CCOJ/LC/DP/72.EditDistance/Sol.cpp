#include <vector>
#include <iostream>
#include <iomanip>
using namespace std;

class Solution {
 public:
   int minDistance(string word1, string word2) {
     int m = word1.size();
     int n = word2.size();
     vector<vector<int>> dp (m+1, vector<int>(n+1, 0));

     for (size_t j = 0; j <= n; j++)
       dp[m][j] = n - j;
     for (size_t i = 0; i <= m; i++)
       dp[i][n] = m - i;

     // This is a 2D dynamic programming problem:
     // Marching backwards from (m, n) to (0, 0)
     // For each (i,j):
     //   if word1[i] == word2[j]
     //     dp[i][j] = dp[i+1][j+1]
     //   else
     //     there're three separate cases to handle
     //     1) Replace character word1[i] with word2[j], and then the problem reduces to converting word1[i+1:end] to word2[j+1:end), which the minimum edit distance is already calculated in dp[i+1][j+1].
     //     2) Remove character word1[i], and try to convert word1[i+1:end] to word2[j:end], which the minimum edit distance is dp[i+1][j]
     //     3) Remove character word2[j], and try to convert word1[i:end] to word2[j+1:end], which the minimum edit distance is dp[i][j+1] 
     for (int i = m-1; i >= 0; i--) {
       for (int j = n-1; j >= 0; j--) {
         if (word1[i] == word2[j]) {
           dp[i][j] = dp[i+1][j+1];
         } else {
           dp[i][j] = min(dp[i+1][j+1], min(dp[i+1][j], dp[i][j+1])) + 1;
         }
       }
     }
     /* printVector(dp, word1, word2); */
     return dp[0][0];
   }

   void printVector(const vector<vector<int>> &V, string w1, string w2) {
     std::cout << "\n";
     for (auto c : w2)
         std::cout << setw(3) << c << ",";
     int w1Index = 0;
     for (auto &v : V) {
       std::cout << "\n";
       for (auto &x : v) {
         std::cout << setw(3) << x << ",";
       }
     std::cout << " " << w1[w1Index++];
     }
     std::cout << "\n";
   }
};


int main() {
  auto sol = Solution();
  /* auto ans = sol.minDistance("horse", "ros"); */
  auto ans = sol.minDistance("zoologicoarcha", "zoog");
  return ans;
}
