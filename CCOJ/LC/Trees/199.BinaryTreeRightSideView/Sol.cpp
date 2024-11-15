#include <vector>
#include <deque>
#include <map>
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
    vector<int> rightSideView(TreeNode* root) {
      if (root == nullptr)
        return {};

      deque<TreeNode *> Q;
      color[root] = 1;
      depth[root] = 0;
      Q.push_back(root);
      while (!Q.empty()) {
        auto *u = Q.front();
        Q.pop_front();
        if (u->left != nullptr && color.find(u->left) == color.end()) {
          color[u->left] = 1;
          depth[u->left] = depth[u] + 1;
          Q.push_back(u->left);
       }

        if (u->right != nullptr && color.find(u->right) == color.end()) {
          color[u->right] = 1;
          depth[u->right] = depth[u] + 1;
          Q.push_back(u->right);
        }
      }

      vector<int> ans;
      map<int, TreeNode *> temp;
      for (auto &[n,d] : depth) {
        temp[d] = n;
      }

      for (auto &[d, n] : temp)
        ans.push_back(n->val);
      //map<int, int> visitedDepth;
      //for (auto I = depth.begin(); I != depth.end(); I++) {
      //  auto &[n0, d0] = *I;
      //  if (visitedDepth.find(d0) == visitedDepth.end()) {
      //    visitedDepth[d0] = 1;
      //    ans.push_back(n0->val);
      //  }
      //}

      //reverse(ans.begin(), ans.end());
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
  auto ans = sol.rightSideView(root);
  return 0;
}
