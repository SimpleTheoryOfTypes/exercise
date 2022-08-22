class Solution {
  vector<vector<TreeNode*>> g;
    map<int, TreeNode *> id2node;
    map<TreeNode *, int> node2id;
    int count = 0;
    map<TreeNode *, int> color;
    map<TreeNode *, int> discover;
public:
    int amountOfTime(TreeNode* root, int start) {
      if (!root->left && !root->right)
        return 0;
      int nNodes = countNodes(root); 
      g = vector<vector<TreeNode *>>(nNodes, vector<TreeNode *>());
      dfs(root);// construct g.
      
      TreeNode *target = nullptr;
      findStart(root, start, target);

      for (const auto &[n, id] : node2id) {
        if (n != target) {
          color[n] = 0;
          discover[n] = INT_MAX;
        }
      }

      discover[target] = 0;
      color[target] = 1;

      queue<TreeNode *> Q;
      Q.push(target);
      while (!Q.empty()) {
        const auto u = Q.front();
        Q.pop();

        for (const auto &nbr : g[node2id[u]]) {
          if (color[nbr] == 0) {
            discover[nbr] = discover[u] + 1;
            color[nbr] = 1;
            Q.push(nbr);
          }
        }
        color[u] = 2;
      }
      
      // find max discover time:
      int ans = INT_MIN;
      for (const auto &[x, d] : discover) {
        if (ans < d)
          ans = d;
      }
      
      return ans;
    }

    // Find start tree node from its value.
    void findStart(TreeNode *root, int start, TreeNode *&target) {
      if (root == nullptr)
        return; 

      if (root->val == start) {
        target = root;
        return;
      }

      findStart(root->left, start, target);
      findStart(root->right, start, target);
    }

    void dfs(TreeNode *node) {
      if (node == nullptr)
        return;

      if (node->left) {
        g[node2id[node]].push_back(node->left);
        g[node2id[node->left]].push_back(node);
      }

      if (node->right) {
        g[node2id[node]].push_back(node->right);
        g[node2id[node->right]].push_back(node);
      }

      dfs(node->left);
      dfs(node->right);
    }

    int countNodes(TreeNode *node) {
      if (node == nullptr)
        return 0;

      id2node[count] = node;
      node2id[node] = count;
      count += 1;

      int numLeft = countNodes(node->left);
      int numRight = countNodes(node->right);
      return numLeft + numRight + 1; 
    }
};
