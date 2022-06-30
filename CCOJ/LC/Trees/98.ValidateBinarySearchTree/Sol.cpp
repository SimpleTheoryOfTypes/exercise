
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
    vector<int> inOrder;
public:
    bool isValidBST(TreeNode* root) {
        dfs(root);
        return isSorted(inOrder);
    }

    // A binary tree is a valid BST iff the inorder traversal is sorted
    // (according to the problem description, it has to be strictly sorted too).
    void dfs(TreeNode* n) {
        if (n == nullptr)
            return;
        dfs(n->left);
        inOrder.push_back(n->val);
        dfs(n->right);
    }

    bool isSorted(vector<int> &v) {
      int x = v[0];
      for (int i = 1; i < v.size(); i++) {
          if (x >= v[i])
              return false;
          x = v[i];
      }
      return true;
    }
};
