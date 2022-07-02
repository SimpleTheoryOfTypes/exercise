#include <map>
#include <vector>
using namespace std;

// Definition for a Node.
class Node {
public:
    int val;
    Node* next;
    Node* random;

    Node(int _val) {
        val = _val;
        next = nullptr;
        random = nullptr;
    }
};

class Solution {
  map<Node *, int> color;
  map<Node *, Node *> old2new;
public:
    Node* copyRandomList(Node* head) {
      dfs(head);

      for (const auto &[Old, New] : old2new) {
        if (Old->next)
          New->next = old2new[Old->next];
        else
          New->next = nullptr;

        if (Old->random)
          New->random = old2new[Old->random];
        else
          New->random = nullptr;
      }

      return old2new[head];
    }

    void dfs(Node *n) {
      if (n == nullptr)
        return;

      color[n] = 1;
      Node *newNode = new Node(n->val); 
      old2new[n] = newNode;
      if (color[n->next] == 0)
        dfs(n->next);
      color[n] = 2;
    }
};

int main() {
  auto sol = Solution();
  auto ans = sol.copyRandomList(nullptr);
  return 0;
}
