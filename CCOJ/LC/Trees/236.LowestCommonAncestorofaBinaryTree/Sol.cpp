class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
      vector<TreeNode *> pathTop;
      vector<TreeNode *> pathToq;
      vector<TreeNode *> path;

      path.clear();
      dfs(root, p, path, pathTop);

      path.clear();
      dfs(root, q, path, pathToq);

      int lcaIndex = -1;
      for (int i = 0, j = 0; i < pathTop.size() && j < pathToq.size(); i++, j++) {
        if (pathTop[i] == pathToq[j]) {
          lcaIndex = i;
        }
      }
      return pathToq[lcaIndex < 0 ? 0 : lcaIndex];
    }

    void dfs(TreeNode *n, TreeNode *target, vector<TreeNode *> &path, vector<TreeNode *> &foundPath) {
      if (n == nullptr)
        return;

      path.push_back(n);
      if (n == target) {
        foundPath = path;
        return;
      }

      dfs(n->left, target, path, foundPath);
      path.erase(path.end() - 1);

      path.push_back(n);
      dfs(n->right, target, path, foundPath);
      path.erase(path.end() - 1);
    }
};
