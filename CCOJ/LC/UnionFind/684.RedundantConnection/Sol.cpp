#include <vector>
#include <map>
#include <iostream>
#include <numeric>
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

// Starting with a graph with all n nodes, zero edges. Then adding edges to the graph and do union-find.
// Whenever we find that adding an edge e will make the graph connecting two nodes that are already
// connected (i.e., their parents are the same node). And that edge is a redundant edge.
class Solution {
  map<int, Node*> id2node;
public:
    vector<int> findRedundantConnection(const vector<vector<int>>& edges) {
      vector<int> ans;

      // Number of nodes in the graph equals the number of edges.
      for (size_t i = 0; i < edges.size(); i++) {
        Node *n = new Node();
        n->id = i + 1;// nodes in the input are 0-indexed.
        id2node[n->id] = n;
        MakeSet(n);
      }

      vector<vector<int>> redundantEdges;
      for (size_t i = 0; i < edges.size(); i++) {
        auto edge = edges[i];
        Node *x = id2node[edge[0]]; 
        Node *y = id2node[edge[1]]; 

        if (FindSet(x) == FindSet(y)) {
          redundantEdges.push_back(edge);
        }

        Union(x, y);
      }

      return redundantEdges.back();
    }
};

int main() {
  auto sol = Solution();
  auto ans = sol.findRedundantConnection({{1,2},{1,3},{2,3}});
  std::cout << "(SimpleTheoryOfTypes) ans = (" << ans[0] << ", " << ans[1] << ")" << std::endl;
  return 0;
}
