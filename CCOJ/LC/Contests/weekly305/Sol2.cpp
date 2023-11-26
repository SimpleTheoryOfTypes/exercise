class Solution {
    int ans = 1; 
    map<int, int> mp;// put restricted nodes in a map
    map<int, int> color;
    int time = 0;
public:
    int reachableNodes(int n, vector<vector<int>>& edges, vector<int>& restricted) {
      vector<vector<int>> dag = vector<vector<int>>(n, vector<int>());
      for (int i = 0; i < n; i++)
        color[i] = 0;
      for (const auto &e : edges) {
        dag[e[0]].push_back(e[1]);
        dag[e[1]].push_back(e[0]);
      }
      
      for (const auto &r : restricted) {
        mp[r] = 1;
      }
        
      dfs(dag, 0);
      return ans;
    }
  
    void dfs(const vector<vector<int>> &dag, int node) {
      color[node] = 1;
      time += 1;
      for (const auto &nbr : dag[node]) {
        if (color[nbr] == 0 && mp.find(nbr) == mp.end()) {
          ans += 1;
          dfs(dag, nbr);
        }
      }
      
      color[node] = 2;
      time += 1;
    }
  
};
