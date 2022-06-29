#include <vector>
using namespace std;

struct TreeNode {
  int val;
  TreeNode *left;
  TreeNode *right;
  TreeNode() : val(0), left(nullptr), right(nullptr) {}
  TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
  TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

class Solution {
  vector<int> depths;
  int diameter = 0;
  public:
    int diameterOfBinaryTree(TreeNode* root) {
      dfs(root);
      return diameter;
    }

    int dfs(TreeNode *node) {
      if (!node)
        return 0;

      int left = dfs(node->left);
      int right = dfs(node->right);

      diameter = max(diameter, left + right);
      return max(left, right) + 1;
    }
};
