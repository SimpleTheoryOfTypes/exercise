#include <map>
#include <iostream>
using namespace std;

struct Node {
  Node *prev;
  Node *next;
  int key;
  int value;
  Node (int k, int v) {
    key = k; value = v; prev = nullptr; next = nullptr; 
  }
};

void print(Node *n) {
  while (n) {
    std::cout << "(" << n->key << "," << n->value << "),";
    n = n->next;
  }
  std::cout << "\n";
}

class LRUCache {
    int cap = -1;
    int count = 0;
    map<int, Node*> mp;
public:
    Node *head;
    Node *tail;

    LRUCache(int capacity) {
      cap = capacity;
      head = new Node(INT_MIN, INT_MIN);// dummy head pointing to the starting node.
      tail = new Node(INT_MIN, INT_MIN);// dummy tail whose predecessor is the end node.
      head->next = tail;
      tail->prev = head;
    }

    void moveToTail(Node *m) {
      Node *pred = m->prev;
      Node *succ = m->next;

      if (succ == tail)
        return;

      pred->next = m->next;
      succ->prev = pred;
      m->next = tail;
      m->prev = tail->prev;
      tail->prev->next = m;
      tail->prev = m;
    }

    int get(int key) {
      if (mp.find(key) == mp.end())
        return -1;

      // Move the key-value pair to the most recently used pair.
      Node *m = mp[key];
      moveToTail(m);
      return m->value;
    }

    void put(int key, int value) {
      // case 1: key already exists, we update its value.
      if (mp.find(key) != mp.end()) {
        mp[key]->value = value;

        // Move the key-value pair to the most recently used pair.
        Node *m = mp[key];
        /* ::print(head); */
        moveToTail(m);
        return;
      }

      // case 2: key does not in the cache, we insert it.
      Node *n = new Node(key, value);
      mp[key] = n; 
      n->key = key;
      count += 1;

      if (count > cap) {
        // need to evict node x
        assert(head->next);
        Node *x = mp[head->next->key];
        Node *lru_p = head->next;//(invariant) head->next always points to the lru node.
        head->next = lru_p->next;
        if (lru_p->next)
          lru_p->next->prev = head;
        x->prev = nullptr;
        x->next = nullptr;
        int keyToEvict = lru_p->key;
        mp.erase(keyToEvict);
      }

      tail->prev->next = n;
      n->prev = tail->prev;
      n->next = tail;
      tail->prev = n;
    }
};

int main() {
  LRUCache* obj = new LRUCache(2);
  obj->put(2,1);
  print(obj->head);
  obj->put(2,2);
  print(obj->head);
  obj->put(1,1);
  print(obj->head);
  obj->put(4,4);
  print(obj->head);
  return obj->get(4);
}
