
class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        auto comp = [] (const pair<int, int> &x, const pair<int, int> &y) {
          return x.second < y.second;
        };
        priority_queue<pair<int, int>, vector<pair<int,int>>, decltype(comp)> pq(comp);
        for (int i = 0; i < nums.size(); i++) {
            pq.push(make_pair(i, nums[i]));
        }

        int count = 0;
        pair<int, int> ans;
        while (!pq.empty()) {
            auto u = pq.top();
            pq.pop();
            count += 1;
            if (count == k) {
                ans = u;
            }
        }
        return ans.second;
    }
};
