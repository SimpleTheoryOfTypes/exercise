#include <vector>
#include <iostream>
#include <string>
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
    TreeNode* buildTreeImpl(const vector<int> &preorder, int pb, int pe, const vector<int> &inorder, int ib, int ie) {
        assert(!preorder.empty() && preorder.size() == inorder.size());
        assert(pb >= 0 && pe < preorder.size());
        assert(ib >= 0 && ie < inorder.size());

        if (pb == pe) {
            assert(ib == ie);
            TreeNode *node = new TreeNode();
            node->val = inorder[ib];
            return node;
        }

        int rootValue = preorder[pb];
        int rootIdx = -1;// root index in inorder.
        for (int i = ib; i <= ie; i++) {
            if (inorder[i] == rootValue) {
                rootIdx = i;
                break;
            }
        }

        TreeNode *root = new TreeNode();
        root->val = rootValue;

        int inLeftStart = ib;
        int inLeftEnd = rootIdx - 1;
        int inRightStart = rootIdx + 1;
        int inRightEnd = ie; 

        bool hasALeftTree = (inLeftStart <= inLeftEnd);

        int preLeftStart = -1; 
        int preLeftEnd = -1; 
        if (hasALeftTree) {
          // Has a left tree:
          preLeftStart = pb + 1;
          preLeftEnd = preLeftStart + (inLeftEnd - inLeftStart);
          root->left = buildTreeImpl(preorder, preLeftStart, preLeftEnd, inorder, inLeftStart, inLeftEnd);
        }

        if (inRightStart <= inRightEnd) {
          // Has a right tree:
          int preRightStart = hasALeftTree ? preLeftEnd + 1 : pb + 1;
          int preRightEnd = preRightStart + (inRightEnd - inRightStart);
          root->right = buildTreeImpl(preorder, preRightStart, preRightEnd, inorder, inRightStart, inRightEnd);
        }
        return root;
    }

    TreeNode* buildTree(const vector<int>& preorder, const vector<int>& inorder) {
        if (preorder.empty())
          return nullptr;

        TreeNode *root = buildTreeImpl(preorder, 0, preorder.size() - 1, inorder, 0, inorder.size() - 1);
        return root;
    }

    void printTree(TreeNode *n, int level = 1) {
      std::string indentStr(level * 2, '-');
      std::cout << indentStr << n->val << "\n";
      level++;
      if (n->left)
        printTree(n->left, level);
      if (n->right)
        printTree(n->right, level);
    }
};

int main() {
  auto sol = Solution();
  auto root = sol.buildTree({3,9,20,15,7}, {9,3,15,20,7});
  /* auto root = sol.buildTree({20,15,7}, {15,20,7}); */
  /* auto root = sol.buildTree({1,2}, {2,1}); */
  /* auto root = sol.buildTree({1,2}, {1,2}); */
  /* auto root = sol.buildTree({1,2,3,4}, {1,2,3,4}); */
  sol.printTree(root);
  return 0;
}
