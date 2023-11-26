class Solution {
public:
    int arithmeticTriplets(vector<int>& nums, int diff) {
      map<int, int> mp;
      for (int i = 0; i < nums.size(); i++) {
        mp[nums[i]] = i;
      }
      
      int ans = 0;
      for (int i = 0; i < nums.size(); i++) {
        int nextVal = nums[i] + diff;
        if (mp.find(nextVal) != mp.end()) {
          int nextVal2 = nextVal + diff;
          if (mp.find(nextVal2) != mp.end())
            ans += 1;
        }
      }
      
      return ans;  
    }
};
