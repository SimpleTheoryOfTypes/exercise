#include <vector>
#include <map>
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
    long long choose(long long n, long long k) {
      if (k == 0)
        return 1;
      return (n / k) * choose(n-1, k-1);
    }

    long long countPairs(int n, const vector<vector<int>>& edges) {
      if (edges.empty()) {
        return choose(n, 2);
      }

      for (int i = 0; i < n; i++) {
        Node *n = new Node();
        MakeSet(n);
        n->id = i;
        id2node[i] = n;
      }

      for (auto &e : edges) {
        assert(e.size() == 2);
        auto &from = e[0];
        auto &to = e[1];
        if (FindSet(id2node[from]) != FindSet(id2node[to])) {
          Union(id2node[from], id2node[to]);
        }
      }

      map<Node*, long long> setSize;
      for (int i = 0; i < n; i++) {
        Node *parent = FindSet(id2node[i]);
        setSize[parent] += 1;
      }

      // Once we know the number of nodes in each group, we simply need to find sum of pairwise multiplications. Number of pairs: N * (N - 1) / 2, hence Complexity (N^2). However, this can be done in linear time with the identity we know:
      // (a + b + c)^2 = (a^2 + b^2 + c^2) + 2*(ab + bc + ca).
      long long sum = 0;
      long long sum2 = 0;
      for (auto &[ignore, count] : setSize) {
        sum += count;
        sum2 += count * count;
      }

      long long ans = 0;
      ans = (sum * sum - sum2) / 2;
      return ans;
    }
};

int main() {
  auto sol = Solution();
  auto ans = sol.countPairs(7, {{0,2},{0,5},{2,4},{1,6},{5,4}});
  return ans;
}
