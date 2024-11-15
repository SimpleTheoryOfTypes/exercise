#include <vector>
using namespace std;

class Solution {
  int theKth = -1;
  bool Found = false;
  int count = 0;
public:
    int kthSmallest(TreeNode* root, int k) {
      dfs(root, k);  
      return theKth;
    }

    void dfs(TreeNode *n, const int k) {
      if (n == nullptr || Found)
        return;
      dfs(n->left, k);
      count += 1;
      if (count == k) {
        theKth = n->val;
        Found = true;
      }
      dfs(n->right, k);
    }
};
