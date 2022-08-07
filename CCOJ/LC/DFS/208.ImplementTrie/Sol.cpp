#include <string>
#include <map>
#include <iostream>
using namespace std;


class TrieNode {
public:
  std::map<char, TrieNode*> children;
  bool isWord;
};

class Trie {
public:
    TrieNode *root;

    Trie() {
      root = new TrieNode();
    }

    //FIXME: add a destructor.

    void dump(TrieNode *r, int level) {
      for (auto &[c, n] : r->children) {
        std::cout << std::string(level, '_')  << c << ": " << n->isWord << std::endl;
        dump(n, level+1);
      }
    }

    void insert(string word) {
      TrieNode *curr = root;
      for (size_t i = 0; i < word.size(); i++) {
        auto w = word[i];
        if (curr->children.find(w) == curr->children.end()) {
          curr->children[w] = new TrieNode();
        }

        curr = curr->children[w];
        if (!curr->isWord) // already a word then don't reset.
          curr->isWord = (i == word.size() - 1) ? true : false;
      }
    }

    bool search(string word) {
      TrieNode *curr = root;
      for (size_t i = 0; i < word.size(); i++) {
        auto w = word[i];
        if (curr->children.find(w) == curr->children.end()) {
          return false;
        }

        curr = curr->children[w];
      }
      return curr->isWord;
    }

    bool startsWith(string prefix) {
      bool ans = true;
      TrieNode *curr = root;
      for (auto w : prefix) {
        if (curr->children.find(w) == curr->children.end()) {
          ans = false;
          break;
        }

        curr = curr->children[w];
      }
      return ans;
    }
};

/**
 * Your Trie object will be instantiated and called as such:
 * Trie* obj = new Trie();
 * obj->insert(word);
 * bool param_2 = obj->search(word);
 * bool param_3 = obj->startsWith(prefix);
 */

/* ["Trie","insert","insert","insert","insert","insert","insert","search","search","search","search","search","search","search","search","search","startsWith","startsWith","startsWith","startsWith","startsWith","startsWith","startsWith","startsWith","startsWith"] */
/* [[],["app"],["apple"],["beer"],["add"],["jam"],["rental"],["apps"],["app"],["ad"],["applepie"],["rest"],["jan"],["rent"],["beer"],["jam"],["apps"],["app"],["ad"],["applepie"],["rest"],["jan"],["rent"],["beer"],["jam"]] */
int main() {
  Trie *trie = new Trie();
  trie->insert("app");
  trie->insert("apple");
  trie->insert("beer");
  trie->insert("add");
  trie->insert("jam");
  trie->insert("rental");
  trie->search("apps");
  trie->dump(trie->root, 0);
  assert(trie->search("app") == true);
  return 0;
}
