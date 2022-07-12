#include <vector>
#include <stack>
#include <string>
using namespace std;

class Solution {
  std::stack<int> stk;
  int base = 0;
public:
    int evalRPN(const vector<string>& tokens) {
      if (tokens.empty())
        return 0;
      while (base != tokens.size()) {
        if (isOperator(tokens[base])) {
          auto rhs = stk.top(); stk.pop();
          auto lhs = stk.top(); stk.pop();
          stk.push(operate(lhs, rhs, tokens[base]));
          base += 1;
          continue;
        }
     
        stk.push(stoi(tokens[base]));
        base += 1;
      }

      return stk.top();
    }

    int operate(int lhs, int rhs, string op) {
      if (op == "*")
        return lhs * rhs;
      else if (op == "/")
        return lhs / rhs;
      else if (op == "+")
        return lhs + rhs;

      return lhs - rhs;
    }

    bool isOperator(string x) {
      if (x.size() == 1 && (x == "*" || x == "+" || x == "-" || x == "/"))
        return true;
      return false;
    }
};

int main() {
  auto sol = Solution();
  auto ans = sol.evalRPN({"2","1","+","3","*"});
  return ans;
}
