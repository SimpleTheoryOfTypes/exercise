#include <vector>
#include <string>
#include <iostream>
using namespace std;

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

class Codec {
    vector<string> s;
public:

    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
      dfs(root);
      string ans = "";
      for (auto &x : s) {
        ans += x;
        ans += ",";
      }
      return ans;
    }

    vector<string> str2vec(string data) {
      vector<string> ans = {};
      if (data.size() == 0)
        return ans;

      int start = 0;
      for (int i = 0; i < data.size(); i++) {
        if (data[i] == ',') {
          ans.push_back(data.substr(start, i - start));
          start = i + 1;
        }
      }
      return ans;
    }

    void dfs(TreeNode *n) {
      if (n == nullptr) {
        s.push_back("null");
        return;
      }

      s.push_back(to_string(n->val));
      dfs(n->left);
      dfs(n->right);
    }

    // Decodes your encoded data to tree.
    TreeNode* deserialize(string data) {
      int index = 0;
      s = str2vec(data);
      auto *root = construct(data, index);
      return root;
    }

    // [1,2,null,null,3,4,5]
    TreeNode* construct(string &data, int &index) {
      assert(index < s.size());
      if (s[index] == "null")
        return nullptr;

      TreeNode *node = new TreeNode(std::stoi(s[index]));
      index += 1;
      node->left = construct(data, index);

      index += 1;
      node->right = construct(data, index);

      return node;
    }
};

TreeNode* buildTree() {
  TreeNode *node1 = new TreeNode(1);
  TreeNode *node2 = new TreeNode(2);
  TreeNode *node3 = new TreeNode(3);
  TreeNode *node4 = new TreeNode(4);

  node1->left = node2;
  node2->left = node3;
  node3->left = node4;
  return node1;
}

TreeNode* buildTree2() {
  TreeNode *node1 = new TreeNode(1);
  TreeNode *node2 = new TreeNode(2);
  TreeNode *node3 = new TreeNode(3);
  TreeNode *node4 = new TreeNode(4);
  TreeNode *node5 = new TreeNode(5);

  node1->left = node2;
  node1->right = node3;
  node3->left = node4;
  node3->right = node5;
  return node1;
}

int main() {
  auto sol = Codec();
  auto *root = buildTree2();
  /* auto ans = sol.serialize(root); */
  //std::cout << "(SimpleTheoryOfTypes) ans = " << ans << std::endl;
  auto ans = sol.deserialize(sol.serialize(root));
  std::cout << "(SimpleTheoryOfTypes) ans = " << ans << std::endl;
  return 0;
}
