/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
      ListNode *slow = head;
      ListNode *fast = head;
      while(true) {
        if (slow == nullptr || fast == nullptr)
          return nullptr;
        
        slow = move(slow);
        fast = move(move(fast));
        
        if (slow == fast)
          break;
        
        //cout << "fast = " << fast->val << "\n";
        //cout << "slow = " << slow->val << "\n";
      }
      
      // Now, we've found out that slow == fast is where the fast pointer and the slow pointer
      // merge, and it can be proven mathematically that the distance from head to slow must
      // be equal to the distance from the merge point (i.e., slow == fast) to the node where
      // the cycle begins.
      // See https://www.youtube.com/watch?v=wjYnzkAhcNk&t=808s for the proof.
      ListNode *slow0 = head;
      ListNode *slow1 = slow;//(invariant) slow == fast
      while (true) {
        if (slow0 == slow1)
          return slow0;
        
        slow0 = move(slow0);
        slow1 = move(slow1);  
      }
      
      assert(false && "should never reach here!");
      return nullptr;
    }
  
    // Move from node x to its child. think of it as a fixed point function operating
    // on x as its argument.
    ListNode* move(ListNode *x) {
      if (x == nullptr || x->next == nullptr)
        return nullptr;
      
      return x->next;
    }
};
