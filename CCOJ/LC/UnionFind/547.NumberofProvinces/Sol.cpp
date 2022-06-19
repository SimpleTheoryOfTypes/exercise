#include <vector>
#include <iostream>
#include <map>
#include <set>
using namespace std;

class Node {
public:
  Node *p;//parent
  int rank;
  int id;
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
    map<int, Node*> id2node;
public:
    int findCircleNum(const vector<vector<int>>& isConnected) {
      int ans;
      int numNodes = isConnected.size();
      vector<vector<int>> Adj(numNodes, vector<int> {});
      for (int i = 0; i < numNodes; i++) {
        for (size_t nbr = 0; nbr < isConnected[i].size(); nbr++) {
          if (nbr != i && isConnected[i][nbr] == 1)
            Adj[i].push_back(nbr);
        }
      }

      for (int i = 0; i < numNodes; i++) {
        Node *n = new Node();
        MakeSet(n);
        id2node[i] = n;
      }

      for (int i = 0; i < numNodes; i++) {
        Node *x = id2node[i];
        for (auto &nbr : Adj[i]) {
          Node *y = id2node[nbr];
          if (FindSet(x) != FindSet(y)) {
            Union(FindSet(x), FindSet(y));
          }
        }
      }

      // Count how many disjoint sets
      set<Node *> uniqueRoots;
      for (int i = 0; i < numNodes; i++) {
        Node *x = id2node[i];
        Node *root = FindSet(x);
        if (uniqueRoots.find(root) == uniqueRoots.end()) {
          uniqueRoots.insert(root);
        }
      }

      return uniqueRoots.size();
    }
};

int main() {
  auto sol = Solution();
  auto ans = sol.findCircleNum({{1,1,0},{0,1,0},{0,0,1}});
  return ans;
}
