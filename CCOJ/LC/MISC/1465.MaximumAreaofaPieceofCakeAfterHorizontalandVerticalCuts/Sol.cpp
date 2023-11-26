#include <vector>
#include <cmath>
#include <algorithm>
using namespace std;

class Solution {
public:
    int maxArea(int h, int w, vector<int>& horizontalCuts, vector<int>& verticalCuts) {
        if (h == 0 || w == 0)
            return 0;

        horizontalCuts.push_back(h);
        horizontalCuts.push_back(0);
        verticalCuts.push_back(w);
        verticalCuts.push_back(0);
        sort(horizontalCuts.begin(), horizontalCuts.end());
        sort(verticalCuts.begin(), verticalCuts.end());

        long long maxHorizontalGap = 0;
        for (int i = 1; i < horizontalCuts.size(); i++) {
                maxHorizontalGap = max<long long>(maxHorizontalGap, horizontalCuts[i] - horizontalCuts[i-1]);
        }

        long long maxVerticalGap = INT_MIN;
        for (int i = 1; i < verticalCuts.size(); i++) {
                maxVerticalGap = max<long long>(maxVerticalGap, verticalCuts[i] - verticalCuts[i-1]);
        }

        return (maxVerticalGap * maxHorizontalGap) % (1000000000 + 7);
    }
};

int main() {
  auto sol = Solution();
  vector<int> hc({2});
  vector<int> vc({2});
  auto ans = sol.maxArea(1000000000, 1000000000, hc, vc);
  return ans;
}



