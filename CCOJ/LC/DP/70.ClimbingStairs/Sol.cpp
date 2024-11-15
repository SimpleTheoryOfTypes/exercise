#include <vector>
using namespace std;

class Solution {
public:
    int climbStairs(int n) {
      if (n <= 2)
        return n;

      vector<int> steps(n+1,0);
      steps[0] = 0;
      steps[1] = 1;
      steps[2] = 2;
      for (int i = 3; i <= n; i++)
        steps[i] = steps[i-1] + steps[i-2];

      return steps[n];
    }
};

int main() {
  auto sol = Solution();
  auto ans = sol.climbStairs(5);
  return ans;
}
