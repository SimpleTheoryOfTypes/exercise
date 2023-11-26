#include <string>
#include <iostream>
#include <vector>
#include <set>
using namespace std;

string solution(string &S) {
    // write your code in C++14 (g++ 6.2.0)
    if (S.size() <= 1)
      return S;

    bool changed = true;
    while (changed && !S.empty()) {
      std::set<size_t> index2Remove;
      for (size_t i = 0; i < S.size() - 1; i++) {
        if ((S[i] == 'A' && S[i+1] == 'B') ||
            (S[i] == 'B' && S[i+1] == 'A') ||
            (S[i] == 'C' && S[i+1] == 'D') ||
            (S[i] == 'D' && S[i+1] == 'C')) {
          index2Remove.insert(i);
          index2Remove.insert(i+1);
        }
      }

      changed = index2Remove.empty() ? false : true;
      string newS;
      for (size_t i = 0; i < S.size(); i++) {
        if (index2Remove.find(i) == index2Remove.end())
          newS += S[i];
      }

      S = newS;
    }

    return S;
}

int main() {
  std::string S = "CABABD";
  auto ans = solution(S);
  std::cout << "(SimpleTheoryOfTypes) ans = " << ans << std::endl;
  return 0;
}
