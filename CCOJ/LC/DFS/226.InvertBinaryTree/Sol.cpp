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
  public:
    TreeNode* invertTree(TreeNode* root) {
      if (!root)
        return root;

      dfs(root);
      return root;
    }

    void dfs(TreeNode *node) {
      TreeNode *temp = node->left;
      node->left = node->right;
      node->right = temp;

      if (node->left)
        dfs(node->left);

      if (node->right)
        dfs(node->right);
    }
};


int main() {
  auto sol = Solution();
  /* auto ans = sol.invertTree(root); */
  return 0;
}

