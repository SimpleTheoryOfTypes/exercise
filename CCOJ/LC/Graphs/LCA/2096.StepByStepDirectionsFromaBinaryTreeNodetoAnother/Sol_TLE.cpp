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
    map<TreeNode *, pair<TreeNode *, int>> parents;// child -> parent (and whether child is left or right);
    map<int, TreeNode*> val2node;
public:
    string getDirections(TreeNode* root, int startValue, int destValue) {
      dfs(root);
      auto *node0 = val2node[startValue];
      auto *node1 = val2node[destValue];
      auto *lca = LCA(root, node0, node1);
      
      string ans = "";
      auto *curr = node0;
      while (true) {
        if (curr == lca) 
          break;
        const auto &x = parents[curr];
        curr = x.first;
        int lr = x.second;// left or right child of parent
        ans += "U";
      }
      
      curr = node1;
      int sz = ans.size();
      while (true) {
        if (curr == lca)
          break;
        const auto &x = parents[curr];
        curr = x.first;
        int lr = x.second;
        if (lr == 0)
          ans.insert(sz, "L");
        else if (lr == 1)
          ans.insert(sz, "R");
        else
          assert(false);
      }
      return ans;
    }
  
    TreeNode* LCA(TreeNode* root, TreeNode * p, TreeNode * q) {
      if(root == NULL || root == p ||  root == q) return root;
      TreeNode* le = LCA(root->left, p ,q);
      TreeNode* ri = LCA(root->right, p, q);
      
      if(le == NULL) return ri;
      else if(ri == NULL ) return le;
      else return root;
    }
  
    void dfs(TreeNode *node) {
      val2node[node->val] = node;
      if (node->left) {
        parents[node->left] = {node, 0};
        dfs(node->left);
      }
      
      if (node->right) {
        parents[node->right] = {node, 1};
        dfs(node->right);
      }
    }
}
