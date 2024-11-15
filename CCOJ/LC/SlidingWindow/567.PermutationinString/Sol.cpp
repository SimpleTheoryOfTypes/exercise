#include <map>
#include <string>
using namespace std;

class Solution {
public:
    bool checkInclusion(const string s1, const string s2) {
      if (s1.size() > s2.size())
        return false; 

      return checkImpl(s1, s2);
    }

    bool checkImpl(const string t, const string s) {
      map<char, int> mp0;
      for (char c = 'a'; c <= 'z'; c++)
        mp0[c] = 0;

      for (const auto &c : t)
        mp0[c] += 1;

      // t <= s in length.
      for (int i = 0; i + t.size() <= s.size(); i++) {
        map<char, int> mp1;
        for (char c = 'a'; c <= 'z'; c++) {
          mp1[c] = 0;
        }

        for (int j = i; j < i + t.size(); j++)
          mp1[s[j]] += 1;

        if (std::equal(mp0.begin(), mp0.end(), mp1.begin()))
          return true;
      }

      return false;
    }
};

int main() {
  auto sol = Solution();
  auto ans = sol.checkInclusion("ab", "a");
  return ans;
}
