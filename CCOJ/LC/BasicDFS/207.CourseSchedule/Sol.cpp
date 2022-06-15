#include <vector>
#include <map>
#include <iostream>
#include <algorithm>
#include <numeric>
using namespace std;

class Solution {
  vector<int> color;
  vector<int> discover;
  vector<int> finish;
  vector<int> parent;
  vector<vector<int>> G;
  int time;
  bool cycleDetected;
public:
  void Constructor(int numCourses) {
    color.resize(numCourses);
    discover.resize(numCourses);
    finish.resize(numCourses);
    parent.resize(numCourses);
    G.resize(numCourses);
    time = 0;
    cycleDetected = false;
  }

  bool canFinish(const int numCourses, const vector<vector<int>>& prerequisites) {
    return !findOrder(numCourses, prerequisites).empty();
  }

  vector<int> findOrder(const int numCourses, const vector<vector<int>>& prerequisites) {
    Constructor(numCourses);

    constructDAG(G, prerequisites);
    for (size_t v = 0; v < G.size(); v++) {
      for (auto n : G[v]) {
        std::cout << "(" << v << "->" << n << ");";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
    DFS(G); 

    // https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
    vector<int> idx(numCourses);
    iota(idx.begin(), idx.end(), 0);
    stable_sort(idx.begin(), idx.end(), [this](size_t i1, size_t i2) {return finish[i1] > finish[i2];});

    if (cycleDetected)
      return vector<int>{};

    for (auto c : idx) {
      std::cout << c << ", ";
    }

    return idx;
  }

  void constructDAG(vector<vector<int>> &G, const vector<vector<int>>& prerequisite) {
    for (auto p : prerequisite) {
      assert(p.size() == 2);
      G[p[1]].push_back(p[0]);
    }
  }

  void DFS(const vector<vector<int>> &G) {
    for (size_t v = 0; v < G.size(); v++) {
      // FIXME: 0: white, 1: gray, 2: black.
      color[v] = 0;
      parent[v] = -1;
    }
    
    for (size_t v = 0; v < G.size(); v++) {
      if (color[v] == 0) {
        DFS_visit(G, v);
      }
    }
  }

  void DFS_visit(const vector<vector<int>> &G, int v) {
    time += 1;
    discover[v] = time;
    color[v] = 1;//GRAY
    for (auto nbr : G[v]) {
      if (color[nbr] == 1) {
        // cycle detected;
        cycleDetected = true;
      }

      if (color[nbr] == 0)
        DFS_visit(G, nbr);
    }
    color[v] = 2;
    time += 1;
    finish[v] = time;
  }
};

int main() {
  int numCourses = 2;
  auto sol = Solution();
  /* auto results = sol.findOrder(numCourses, {{1,0},{2,0},{3,1},{3,2}}); */
  auto results = sol.findOrder(numCourses, {{1,0}});
  return 0;
}

