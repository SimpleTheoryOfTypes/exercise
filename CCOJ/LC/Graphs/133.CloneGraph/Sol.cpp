#include <vector>
#include <iostream>
#include <map>
using namespace std;

// Definition for a Node.
class Node {
public:
    int val;
    vector<Node*> neighbors;
    Node() {
        val = 0;
        neighbors = vector<Node*>();
    }
    Node(int _val) {
        val = _val;
        neighbors = vector<Node*>();
    }
    Node(int _val, vector<Node*> _neighbors) {
        val = _val;
        neighbors = _neighbors;
    }
};

class Solution {
public:
    map<Node *, Node *> new2old;
    map<Node *, Node *> old2new;
    map<Node *, int> color;

    Node* cloneGraph(Node* node) {
      if (!node)
        return nullptr;

      color.clear();
      dfs(node);

      for (auto &[New, Old] : new2old) {
        for (auto &nbr : Old->neighbors) {
          New->neighbors.push_back(old2new[nbr]);
        } 
      }

      Node *ans;
      for (auto &[New, Old] : new2old) {
        if (New->val == 1) { 
          ans = New;
          break;
        }
      }

      color.clear();printDFS(ans);
      return ans;
    }

    void dfs(Node* node) {
      color[node] = 1;
      Node *clone = new Node(node->val);
      new2old[clone] = node;
      old2new[node] = clone;
      std::cout << node->val << "\n";
      for (int i = 0; i < node->neighbors.size(); i++) {
        if (color.find(node->neighbors[i]) == color.end()) {
          dfs(node->neighbors[i]);
        }
      }
    }

    void printDFS(Node *node) {
      //call color.clear()
      color[node] = 1;
      std::cout << node->val << "\n";
      for (int i = 0; i < node->neighbors.size(); i++) {
        if (color.find(node->neighbors[i]) == color.end()) {
          printDFS(node->neighbors[i]);
        }
      }
    }
};

Node* buildGraph() {
  Node *n1 = new Node(1);
  Node *n2 = new Node(2);
  Node *n3 = new Node(3);
  Node *n4 = new Node(4);

  n1->neighbors.push_back(n2);
  n1->neighbors.push_back(n4);
  n2->neighbors.push_back(n1);
  n2->neighbors.push_back(n3);
  n3->neighbors.push_back(n2);
  n3->neighbors.push_back(n4);
  n4->neighbors.push_back(n1);
  n4->neighbors.push_back(n3);
  return n1;
}

int main() {
  Node *g = buildGraph();
  auto sol = Solution();
  auto ans = sol.cloneGraph(g);
  return 0;
}
