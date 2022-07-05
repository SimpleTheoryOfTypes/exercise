#include <vector>
#include <numeric>
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
  vector<TreeNode *> myGoodNodes;
  vector<int> pathSoFar;
public:
    int goodNodes(TreeNode* root) {
      if (root == nullptr)
        return 0;

      dfs(root);
      return myGoodNodes.size();
    }

    void dfs(TreeNode* n) {
      if (n == nullptr)
        return;

      if (pathSoFar.empty() || n->val >= *max_element(pathSoFar.begin(), pathSoFar.end())) {
        myGoodNodes.push_back(n);
      }
      pathSoFar.push_back(n->val);
      dfs(n->left);
      dfs(n->right);
      pathSoFar.erase(pathSoFar.end() - 1);
    }
};

TreeNode* buildTree() {
  TreeNode *node1 = new TreeNode (3);
  TreeNode *node2 = new TreeNode (1);
  TreeNode *node3 = new TreeNode (4);
  TreeNode *node4 = new TreeNode (3);
  TreeNode *node5 = new TreeNode (1);
  TreeNode *node6 = new TreeNode (5);

  node1->left = node2;
  node1->right = node3;
  node2->left = node4;
  node3->left = node5;
  node3->right = node6;

  return node1;
}

int main() {
  auto sol = Solution();
  auto *root = buildTree();
  auto ans = sol.goodNodes(root);
  return ans;
}
