#include <vector>
#include <cstring>
#include <string>
#include <iostream>
using namespace std;
class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> ans;

        string Nstr = "";
        for (int i = 0; i < nums.size(); i++)
            Nstr.push_back('1');

        int N = stoi(Nstr, nullptr, 2);

        while (N > 0) {
          string Nstr = int_binary_to_str(N, nums.size());
          std::cout << "(SimpleTheoryOfTypes) N = " << N << ", Nstr = "<< Nstr << std::endl;
          vector<int> a_subset = {};
          for (int i = 0; i < nums.size(); i++) {
            if (Nstr[i] == '1') {
              a_subset.push_back(nums[i]);
            }
          }
          ans.push_back(a_subset);
          N--;
        }

        ans.push_back(vector<int> {});
        return ans;
        
    }

    void print_vector(vector<int>& v) {
        for (size_t i = 0; i < v.size(); i++)
            std::cout << "|" << v[i] << "\n";
    }

    string int_binary_to_str(unsigned int x, unsigned int nbits) { 
      // convert x into a string of 0 and 1 binary representation in nbits.
      unsigned int mask = 1;
      string result;
      for (size_t i = 0; i < nbits; i++) {
         unsigned int bit_value = x & mask;
         if (bit_value == 0)
             result.insert(0, "0");
         else
             result.insert(0, "1");
         x = x >> 1;
      }
      return result;
    }
};

int main() {
  vector<int> nums {1,2,3};
  auto sol = Solution();
  auto ans = sol.subsets(nums);

  std::cout << stoi("011", nullptr, 2) << "\n";
  std::cout << sol.int_binary_to_str(7, 3) << "\n";


  return 0;
}
