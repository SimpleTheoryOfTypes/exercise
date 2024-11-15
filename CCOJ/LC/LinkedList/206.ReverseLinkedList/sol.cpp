
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
      if (!head || !head->next)
        return head;

      stack<ListNode *> stk;
      while (head) {
        stk.push(head);
        head = head->next;
      }

      ListNode *ans = stk.top();
      while (true) {
        auto n = stk.top();
        stk.pop();
        if (stk.emtpy()) {
          n->next = nullptr;
          break;
        } 
        n->next = stk.top();
      }
        
      return ans;
    }

};
