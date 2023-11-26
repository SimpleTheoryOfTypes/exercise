#include <vector>
#include <map>
#include <iostream>
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
  map<int, int> val2id;
public:
    int longestConsecutive(const vector<int>& nums) {
      int ans;
      if (nums.empty())
        return 0;
      
      for (auto i = 0; i < nums.size(); i++) {
        int x = nums[i];
        Node *n = new Node();
        MakeSet(n);
        n->id = i;
        id2node[i] = n;
        val2id[x] = n->id;
      }

      for (auto &[num, i] : val2id) {
        if (val2id.find(num - 1) != val2id.end()) {
          auto idx = val2id[num - 1];
          if (FindSet(id2node[i]) != FindSet(id2node[idx])) {
            Union(id2node[i], id2node[idx]);
          }
        }
        if (val2id.find(num + 1) != val2id.end()) {
          auto idx = val2id[num + 1];
          if (FindSet(id2node[i]) != FindSet(id2node[idx])) {
            Union(id2node[i], id2node[idx]);
          }
        }
      }

      map<Node*, int> ChildrenCount;
      for (auto i = 0; i < nums.size(); i++) {
        auto *node = id2node[i];
        auto *root = FindSet(node);
        if (ChildrenCount.find(root) != ChildrenCount.end())
          ChildrenCount[root] += 1;
        else
          ChildrenCount[root] = 0;
      }

      int maxChildren = 0;
      for (auto &[parent, numChildren] : ChildrenCount) {
        maxChildren = max(maxChildren, numChildren);
      }

      ans = maxChildren + 1;
      return ans;
    }
};

int main() {
  auto sol = Solution();
  auto ans = sol.longestConsecutive({100,4,200,1,2,3,2});
  return ans;
}
