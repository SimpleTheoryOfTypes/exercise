#include <set>
#include <vector>
#include <iostream>
#include <numeric>
using namespace std;

// Brute force backtracking
//class Solution {
//  vector<int> partialSum;
//  vector<vector<int>> allSolutions;
//  bool Found = false;
//public:
//    int minimumNumbers(int num, int k) {
//      if (num == 0)
//        return 0;
//
//      if (!endInK(num, k) && k == 0)
//        return -1;
//
//      allSolutions = vector<vector<int>> {};
//      partialSum = vector<int> {};
//      backtrack(num, k);
//
//      if (allSolutions.empty())
//        return -1;
//
//      int minLen = INT_MAX;
//      for (auto ans : allSolutions)
//        minLen = min<int>(minLen, ans.size());
//
//      return minLen;
//    }
//
//    bool endInK(int i, int k) {
//      if (i < k)
//        return false;
//
//      if ((i - k) % 10 == 0)
//        return true;
//
//      return false;
//    }
//
//    void printVector(const vector<int> &V) {
//      for (auto &v : V) {
//          std::cout << v << ",";
//      std::cout << std::endl;
//      }
//    }
//
//    void backtrack(int num, int k) {
//      if (accumulate(partialSum.begin(), partialSum.end(), 0) == num) {
//        allSolutions.push_back(partialSum);
//        Found = true;
//        std::cout << "(SimpleTheoryOfTypes) size = " << allSolutions.size() << std::endl;
//        return;
//      }
//
//      if (accumulate(partialSum.begin(), partialSum.end(), 0) > num)
//        return;
//
//      for (int i = num; i >= 0; i--) {
//        if (i == 0 && !partialSum.empty())
//          continue;
//
//        if (Found)
//          return;
//
//        if (endInK(i, k)) {
//          partialSum.push_back(i);
//          printVector(partialSum);
//          backtrack(num, k);
//          partialSum.erase(partialSum.end() - 1);
//          printVector(partialSum);
//        }
//      }
//    }
//};

// Gold:
// 58,9 => [9,49], we have two 9's and a 40.
// Necessary condition is 58 - 2*9 has to divible by 10.
class Solution {
public:
    int minimumNumbers(int num, int k) {
        int i;
        if (num==0) return 0;
        for (i=1;i<=10;i++)
        {
            if ((num>=i*k)&&((num-i*k)%10==0)) return i;
        }
        return -1;
    }
};

int main() {
  auto sol = Solution();
  auto ans = sol.minimumNumbers(4,0);
  std::cout << "(SimpleTheoryOfTypes) ans = " << ans << std::endl;
  return ans;
}

