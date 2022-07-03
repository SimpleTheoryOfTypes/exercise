#include <vector>
#include <deque>
#include <map>
#include <iostream>
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
    map<TreeNode *, int> color;
    map<TreeNode *, int> depth;
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
      vector<vector<int>> ans;
      if (root == nullptr)
        return ans;

      deque<TreeNode *> Q;
      Q.push_back(root);
      color[root] = 1;
      depth[root] = 0;
      while (!Q.empty()) {
        auto *u = Q.front();
        Q.pop_front();
        if (u->left && color.find(u->left) == color.end()) {
          Q.push_back(u->left);
          color[u->left] = 1;
          depth[u->left] = depth[u] + 1;
        }
        if (u->right && color.find(u->right) == color.end()) {
          Q.push_back(u->right);
          color[u->right] = 1;
          depth[u->right] = depth[u] + 1;
        }
        color[u] = 2;//BLACK
      }

      int maxDepth = 0;
      for (const auto &[n, d] : depth) {
        if (d > maxDepth)
          maxDepth = d;
      }

      ans = vector<vector<int>>(maxDepth + 1, vector<int>());
      for (auto I = depth.begin(); I != depth.end(); I++) {
        auto &[n, d] = *I;
        ans[d].push_back(n->val);
      }
      return ans;
    }
};

TreeNode* buildTree() {
  TreeNode *node3 = new TreeNode (3);
  TreeNode *node9 = new TreeNode (9);
  TreeNode *node20 = new TreeNode (20);
  node3->left = node9;
  node3->right = node20;

  TreeNode *node15 = new TreeNode (15);
  TreeNode *node7 = new TreeNode (7);
  node20->left = node15;
  node20->right = node7;

  return node3;
}

int main() {
  auto sol = Solution();
  auto *root = buildTree();
  auto ans = sol.levelOrder(root);
  return 0;
}
