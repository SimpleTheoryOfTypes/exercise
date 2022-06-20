#include <vector>
#include <string>
#include <iostream>
#include <queue>
#include <map>
using namespace std;

int solution(string &S, vector<int> &A) {
  if (A.empty())
    return 0;

  if (A.size() == 1)
    return 1;

  if (A.size() == 2) {
    if (S[0] == S[1])
      return 1;
    else
      return 2;
  }

  int ans = -1;
  int N = A.size();
  vector<vector<int>> Adj(N,  vector<int> {});

  for (size_t i = 0; i < A.size(); i++) {
    if (A[i] == -1)
      continue;

    Adj[A[i]].push_back(i);
    Adj[i].push_back(A[i]);
  }

  // Do BFS from each node.
  int maxPathLen = 0xFFFFFFFF;
  for (size_t n = 0; n < A.size(); n++) {

    map<int, int> color;
    for (size_t v = 0; v < A.size(); v++) {
      if (v != n)
        color[v] = 0;
      else
        color[v] = 1;
    }

    int pathLen = 1;
    queue<int> Q;
    Q.push(n);
    while (!Q.empty()) {
      int u = Q.front(); Q.pop();
      char letter = S[u];
      char desiredLetter = letter == 'a' ? 'b' : 'a';
      for (auto v : Adj[u]) {
        if (color[v] == 0 && S[v] == desiredLetter) {
          color[v] = 1;
          Q.push(v);
          pathLen += 1;
        }
      }
      color[u] = 2;
    }
    maxPathLen = max(maxPathLen, pathLen);
    std::cout << "(SimpleTheoryOfTypes) From " << n << ", pathLen = " << pathLen << std::endl;
  }

  ans = maxPathLen;
  return ans;
}

int main() {
  string S = "abbab";
  vector<int> A = {-1,0,0,0,2};
  auto ans = solution(S, A);
  return ans;
}
