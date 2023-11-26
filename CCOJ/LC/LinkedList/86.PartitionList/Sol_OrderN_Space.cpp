#include <vector>
#include <iostream>
using namespace std;

struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution {
public:
    ListNode* partition(ListNode* head, int x) {
      // O(N) space
      auto *p0 = new ListNode();
      auto *p1 = new ListNode();

      // Auxiliary header trick
      // Input: 1->4->3->2->5->2
      // After partitioning:
      //   head0->1->2->2
      //   head1->4->3->5
      auto *head0 = p0;
      auto *head1 = p1;

      while (head) {
        if (head->val < x) {
          p0->next = new ListNode();
          p0 = p0->next;
          p0->val = head->val;
        } else {
          p1->next = new ListNode();
          p1 = p1->next;
          p1->val = head->val;
        }
        head = head->next;
      }

      p0->next = head1->next;
      return head0->next;
    }
};

void printList(ListNode *head) {
  while (head) {
    std::cout << head->val << "->";
    head = head->next;
  }
  std::cout << "\n";
}

ListNode* buildList() {
  auto* n0 = new ListNode(1);
  auto* n1 = new ListNode(4);
  auto* n2 = new ListNode(3);
  auto* n3 = new ListNode(2);
  auto* n4 = new ListNode(5);
  auto* n5 = new ListNode(2);
  n0->next = n1;
  n1->next = n2;
  n2->next = n3;
  n3->next = n4;
  n4->next = n5;
  return n0;
}

int main() {
  auto sol = Solution();
  printList(buildList());
  auto ans = sol.partition(buildList(), 3);
  printList(ans);
  return 0;
}
