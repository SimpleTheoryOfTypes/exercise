#include <vector>
#include <iostream>
#include <map>
using namespace std;

struct TreeNode {
  int val;
  TreeNode *left;
  TreeNode *right;
  TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

class Solution {
  int index = 0;
  map<TreeNode*, int> tourIndex;
  map<int, TreeNode*> tourIndex2Node;
  map<int, int> depth;
  public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
      if (!root)
        return nullptr;

      dfs(root, 0);
      TreeNode *ans;

      int startIndex = min(tourIndex[p], tourIndex[q]);
      int endIndex = max(tourIndex[p], tourIndex[q]);
      int minDepth = INT_MAX;
      int minDepthIndex = -1;
      for (int i = startIndex; i <= endIndex; i++) {
        if (minDepth > depth[i]) {
          minDepth = depth[i];
          minDepthIndex = i;
        }
      }
      
      ans = tourIndex2Node[minDepthIndex];
      return ans;
    }

    // Euler tour: https://www.youtube.com/watch?v=sD1IoalFomA
    void dfs(TreeNode *n, int Depth) {
      if (n == nullptr)
        return;

      visit(n, Depth);
      if (n->left) {
        dfs(n->left, Depth + 1);
        visit(n, Depth);
      }

      if (n->right) {
        dfs(n->right, Depth + 1);
        visit(n, Depth);
      }
    }

    void visit(TreeNode *n, int Depth) {
      tourIndex[n] = index;
      tourIndex2Node[index] = n;
      depth[index] = Depth;
      index += 1;
    }

};

TreeNode* buildTree() {
  TreeNode* node1 = new TreeNode(1);
  TreeNode* node2 = new TreeNode(2);
  TreeNode* node3 = new TreeNode(3);
  TreeNode* node4 = new TreeNode(4);

  node1->left = node2;
  node1->right = node3;
  node2->right = node4;

  return node1;
}

int main() {
  auto sol = Solution();
  auto *root = buildTree();
  auto ans = sol.lowestCommonAncestor(root, root, root);
  return 0;
}
