#include <vector>
#include <iostream>
#include <map>
#include <queue>
using namespace std;

struct TreeNode {
  int val;
  TreeNode *left;
  TreeNode *right;
  TreeNode() : val(0), left(nullptr), right(nullptr) {}
  TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
  TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};


class Node {
public:
  Node *p;//parent
  int rank;
  TreeNode *tnp;//Tree node pointer
};

void MakeSet(Node *x) {
  x->p = x;
  x->rank = 0;
}

void Link(Node *x, Node *y) {
  if (x->rank > y->rank) {
    y->p = x;
  } else {
    x->p = y;
    if (x->rank == y->rank)
      y->rank = y->rank + 1;
  }
}

Node* FindSet(Node *x) {
  if (x != x->p)
    x->p = FindSet(x->p);
  return x->p;
}

void Union(Node *x, Node *y) {
  Link(FindSet(x), FindSet(y));
}


class Solution {
  pair<TreeNode *, TreeNode *> thePair;
  TreeNode *ans;
  map<TreeNode *, int > color;
  map<TreeNode *, Node*> t2n;
  map<TreeNode *, int> depth;
  int count = 0;
public:
    TreeNode* lcaDeepestLeaves(TreeNode* root) {
      if (root->left == nullptr &&
          root->right == nullptr)
        return root;


      dfs(root);

      int maxDepth = 0;
      for (auto &[tn, d] : depth) {
        //std::cout << "node " << tn->val << " at depth " << d << std::endl;
        maxDepth = max(maxDepth, d);
      }

      vector<TreeNode *> DeepestLeaves;
      for (auto &[tn, d] : depth) {
        if (d == maxDepth)
          DeepestLeaves.push_back(tn);
      }

      if (DeepestLeaves.size() == 1)
        return DeepestLeaves[0];


      int minDepth = INT_MAX;
      TreeNode *lowestCommon = nullptr;
      for (auto &x : DeepestLeaves) {
        for (auto &y : DeepestLeaves) {
          if (x != y) {
            thePair.first = x;
            thePair.second = y;
            lca(root);
            if (depth[ans] < minDepth) {
              minDepth = depth[ans];
              lowestCommon = ans;
            }
          }
        }
      }

      return lowestCommon;
    }

    void dfs(TreeNode *n) {
      if (n == nullptr)
        return;

      depth[n] = count;

      count += 1;
      if (n->left)
        dfs(n->left);
      count -= 1;

      count += 1;
      if (n->right)
        dfs(n->right);
      count -= 1;
    }

    // Tarjan's offline
    void lca(TreeNode* u) {
      if (u == nullptr)
        return;

      Node *node = new Node();
      node->tnp = u;
      t2n[u] = node;
      MakeSet(node);
      FindSet(t2n[u])->tnp = u;

      if (TreeNode *v = u->left) {
        lca(u->left);
        Union(t2n[u], t2n[v]);
        FindSet(t2n[u])->tnp = u;
      }

      if (TreeNode *v = u->right) {
        lca(u->right);
        Union(t2n[u], t2n[v]);
        FindSet(t2n[u])->tnp = u;
      }

      color[u] = 2;//BLACK



      if (u == thePair.first) {
        if (color[thePair.second] == 2) {
          ans = FindSet(t2n[thePair.second])->tnp;
        }
      }

      if (u == thePair.second) {
        if (color[thePair.first] == 2) {
          ans = FindSet(t2n[thePair.first])->tnp;
        }
      }
    }
};

TreeNode* buildTree() {
  TreeNode *node3 = new TreeNode(3);
  TreeNode *node5 = new TreeNode(5);
  TreeNode *node1 = new TreeNode(1);
  TreeNode *node6 = new TreeNode(6);
  TreeNode *node2 = new TreeNode(2);
  TreeNode *node0 = new TreeNode(0);
  TreeNode *node8 = new TreeNode(8);
  TreeNode *node7 = new TreeNode(7);
  TreeNode *node4 = new TreeNode(4);

  node3->left = node5;
  node3->right = node1;

  node5->left = node6;
  node5->right = node2;

  node2->left = node7;
  node2->right = node4;

  node1->left = node0;
  node1->right = node8;

  return node3;
}

TreeNode* buildTree2() {
  TreeNode *node1 = new TreeNode(1);
  return node1;
}

TreeNode* buildTree3() {
  TreeNode *node0 = new TreeNode(0);
  TreeNode *node1 = new TreeNode(1);
  TreeNode *node3 = new TreeNode(3);
  TreeNode *node2 = new TreeNode(2);

  node0->left = node1;
  node0->right = node3;
  node1->right = node2;

  return node0;
}

TreeNode* buildTree4() {
  TreeNode *node1 = new TreeNode(1);
  TreeNode *node3 = new TreeNode(3);
  TreeNode *node2 = new TreeNode(2);

  node1->left = node2;
  node1->right = node3;
  return node1;
}

TreeNode* buildTree5() {
  TreeNode *node1 = new TreeNode(1);
  TreeNode *node2 = new TreeNode(2);
  TreeNode *node3 = new TreeNode(3);
  TreeNode *node4 = new TreeNode(4);
  TreeNode *node5 = new TreeNode(5);

  node1->right = node2;
  node2->right = node3;
  node3->right = node4;
  node4->right = node5;
  return node1;

}

int main() {
  auto sol = Solution();
  TreeNode *root = buildTree2();
  auto ans = sol.lcaDeepestLeaves(root);
  return 0;
}


//class Solution {
//  vector<TreeNode *> inOrder;
//  public:
//
//    unsigned int nextPowerOf2(unsigned int n) {
//      n--;
//      n |= n >> 1;
//      n |= n >> 2;
//      n |= n >> 4;
//      n |= n >> 8;
//      n |= n >> 16;
//      n++;
//      return n;
//    }
//
//    TreeNode* lcaDeepestLeaves(TreeNode* root) {
//      TreeNode *ans;
//
//      // BFS inorder traverse all nodes into a vector
//      inOrder.clear();
//      bfs(root);
//
//      // Round the size of tree nodes to a pow-of-2 number - 1, since my bfs func
//      // might print more null nodes. E.g., a binary tree with depth = 5 has 2^5-1
//      // total nodes.
//      int n2 = nextPowerOf2(inOrder.size())/2 - 1;
//
//      // One node tree => return itself.
//      if (n2 == 1)
//        return inOrder[];
//
//      int numDeepestNodes = 0;
//      TreeNode *singleDeepestNode = nullptr; 
//      for (int i = n2-1; i >= (n2-1)/2; i--) {
//        if (inOrder[i]) {
//          numDeepestNodes += 1;
//          singleDeepestNode = inOrder[i];
//        }
//      }
//
//      if (numDeepestNodes == 1)
//        return singleDeepestNode;
//
//      while (true) {
//        map<int, TreeNode*> parentMap;
//        for (int i = n2-1; i >= (n2-1)/2; i--) {
//          if (inOrder[i]) {
//            std::cout << i << ",";
//            parentMap[(i-1)/2] = inOrder[(i-1)/2];
//          }
//        }
//
//        if (parentMap.size() == 1) {
//          auto &[index, node] = *(parentMap.begin());
//          ans = node;
//          std::cout << "lowest common parent" << ans->val << std::endl;
//          break;
//        }
//
//        n2 = (n2+1)/2 - 1;
//      }
//        
//      return ans;
//    }
//
//    void bfs(TreeNode *n) {
//      queue<TreeNode *> Q;
//      Q.push(n);
//      while (!Q.empty()) {
//        auto *u = Q.front(); Q.pop();
//        inOrder.push_back(u);
//        if (u != nullptr) {
//          Q.push(u->left);
//          Q.push(u->right);
//        }
//      }
//    }
//};
