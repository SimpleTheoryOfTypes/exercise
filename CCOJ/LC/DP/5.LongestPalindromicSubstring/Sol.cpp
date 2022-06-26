#include <string>
#include <iostream>
using namespace std;
class Solution {
public:
    // Reference: https://en.wikipedia.org/wiki/Longest_palindromic_substring
    string longestPalindrome(string s) {
      // Pad string with '|': s = "book", then s2 = "|b|o|o|k|"
      string s2 = "|"; 
      for (int i = 0; i < s.size(); i++) {
        s2 += s[i];
        s2 += "|";
      }
      /* std::cout << "(SimpleTheoryOfTypes) s2 = " << s2 << std::endl; */

      int maxRadius = 0xFFFFFFFF;
      int optimalCenter = -1;
      for (int center = 0; center < s2.size(); center++) {
        int radius = 0;
        while ((center - (radius+1)) >= 0 && (center + (radius+1)) < s2.size() && s2[center-(radius+1)] == s2[center+(radius+1)]) {
          radius += 1;
        }

        if (radius > maxRadius) {
          maxRadius = radius;
          optimalCenter = center;
        }
      }

      assert(maxRadius >= 0);
      assert(optimalCenter != -1);
      /* std::cout << "(SimpleTheoryOfTypes) maxRadius = " << maxRadius << std::endl; */
      /* std::cout << "(SimpleTheoryOfTypes) optimalCenter = " << optimalCenter << std::endl; */
      string s2Result = s2.substr(optimalCenter - maxRadius, 2 * maxRadius + 1);
      string ans;
      for (int i = 0; i < s2Result.size(); i++) {
        if (s2Result[i] != '|')
          ans += s2Result[i];
      }

      /* std::cout << "(SimpleTheoryOfTypes) ans = " << ans << std::endl; */
      return ans;
    }
};

int main() {
  auto sol = Solution();
  /* auto ans = sol.longestPalindrome("book"); */
  auto ans = sol.longestPalindrome("babad");
  return 0;
}
