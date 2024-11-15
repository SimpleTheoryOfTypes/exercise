#include <vector>
#include <map>
#include <numeric>
using namespace std;

class Solution {
  int N;
  int side;
  map<unsigned, bool> cache;
public:
    bool makesquare(const vector<int>& matchsticks) {
      N = matchsticks.size();
      int total = std::accumulate(matchsticks.begin(), matchsticks.end(), 0);
      if (total % 4 != 0)
        return false;

      side = total / 4; 
      return backtrack(3, 0, side, matchsticks);
    }

    bool backtrack(int sides, unsigned mask, int current, const vector<int>& matchsticks) {
      if (sides == 0)
        return true;

      if (current == 0) {
        sides -= 1;
        current = side;
      }

      if (cache.find(mask) != cache.end())
        return cache[mask];

      for (int i = 0; i < N; i++) {
        if ((mask & (1 << i)) == 0 && current >= matchsticks[i]) {
          if (backtrack(sides, mask | (1 << i), current - matchsticks[i], matchsticks)) {
            cache[mask] = true;
            return true;
          }
        }
      }

      cache[mask] = false;
      return false;
    }
};

int main() {
  auto sol = Solution();
  auto ans = sol.makesquare({1,1,2,2,2});
  return ans;
}
