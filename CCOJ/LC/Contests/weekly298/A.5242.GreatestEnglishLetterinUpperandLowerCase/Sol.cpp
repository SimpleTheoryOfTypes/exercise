#include <map>
#include <string>
#include <iostream>
using namespace std;
class Solution {
  map<char, int> upper;
  map<char, int> lower;
public:
    string greatestLetter(string s) {
      string ans = "";
      for (size_t i = 0; i < s.size(); i++) {
        if (s[i] >= 'a' && s[i] <= 'z')
          lower[s[i]] = 1;
        else if (s[i] >= 'A' && s[i] <= 'Z')
          upper[s[i]] = 1;
      }

      for (auto i = 'Z'; i >= 'A'; i--) {
        if (upper.find(i) != upper.end() && lower.find(i + 32) != lower.end()) {
          ans = i;  
          break;
        }
      }
      
      return ans;
    }
};

int main() {
  auto sol = Solution();
  auto ans = sol.greatestLetter("arRAzFif");
  std::cout << "(SimpleTheoryOfTypes) ans = " << ans << std::endl;
  return 0; 
}
