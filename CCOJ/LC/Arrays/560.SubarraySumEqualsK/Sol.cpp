#include <vector>
#include <map>
using namespace std;

class Solution {
public:
    int subarraySum(const vector<int>& nums, int k) {
       map<int, int> mp;

       int prefix = 0;
       mp[prefix] += 1;

       int ans = 0;
       for (int i = 0; i < nums.size(); i++) {
         prefix += nums[i];
         if (mp.find(prefix - k) != mp.end())
           ans += mp[prefix - k];
         mp[prefix] += 1;
       }
      
       return ans;
    }
};

int main() {
  auto sol = Solution();
  /* auto ans = sol.subarraySum({1,2,3}, 3); */
  /* auto ans = sol.subarraySum({1,-1,0}, 0); */
  auto ans = sol.subarraySum({1}, 1);
  return ans;
}
