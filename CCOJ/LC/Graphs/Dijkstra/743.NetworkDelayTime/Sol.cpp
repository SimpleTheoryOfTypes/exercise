class Solution {
    vector<int> d;
    map<pair<int, int>, int> weight;// pair = <from_node, to_node>, weight
    vector<pair<int, int>> Q;// pair<node id, d>
public:
    int networkDelayTime(vector<vector<int>>& times, int n, int k) {
      d = vector<int>(n, INT_MAX);
      int target = k - 1;

      vector<vector<int>> adj(n, vector<int>());
      for (const auto &t : times) {
        const auto u = t[0] - 1; // - 1 b/c the input is 1-indexed, but prefer 0-indexed.
        const auto v = t[1] - 1;
        const auto w = t[2];

        adj[u].push_back(v);
        weight[{u,v}] = w;
      }

      dijkstra(adj, weight, target);
      //printv(d);
      int ans = *max_element(d.begin(), d.end());
      if (ans == INT_MAX)
        return -1;
      return ans;
    }

    void printv(vector<int> V) {
      for (const auto &x : V)
        cout << "d = " << x << ",";
      cout << "\n";
    }

    void dijkstra(vector<vector<int>> &G, map<pair<int, int>, int> &weight, int s) {
      initialize_single_source(G, s);
      set<int> S;
      for (int v = 0; v < G.size(); v++) {
        Q.push_back({v,d[v]});
      }
      make_heap(Q.begin(), Q.end(), [this] (const auto &lhs, const auto &rhs) {return d[lhs.first] > d[rhs.first];});
      //for (const auto &[x, y] : Q)
      //  cout << "x = " << x << ", y = " << y << ",";
      //cout << "\n";

      while (!Q.empty()) {
        const auto [u, ignore] = Q[0];
        pop_heap(Q.begin(), Q.end(), [this] (const auto &lhs, const auto &rhs) {return d[lhs.first] > d[rhs.first];});
        Q.pop_back();

        S.insert(u);
        for (const auto &v : G[u]) {
          relax(u, v, weight);
        }
      }
    }

    void relax(int u, int v, map<pair<int,int>, int> &w) {
      //cout << "d[u] = " << d[u] << "\n";
      //cout << "d[v] = " << d[v] << "\n";
      //cout << "w[u,v] = " << w[{u,v}] << "\n";
      if (d[u] != INT_MAX && d[v] > d[u] + w[{u,v}]) {
        d[v] = d[u] + w[{u,v}];
        make_heap(Q.begin(), Q.end(), [this] (const auto &lhs, const auto &rhs) {return d[lhs.first] > d[rhs.first];});
      }
    }

    void initialize_single_source(vector<vector<int>> &G, int s) {
      for (int v = 0; v < G.size(); v++) {
        d[v] = INT_MAX;
      }
      d[s] = 0;
    }
};
