class Solution {
     vector<vector<int>> dd;
     vector<pair<int,int>> Q;// Q[i] = a vector of three numbers: r, c, dd
public:
    int minimumEffortPath(const vector<vector<int>>& heights) {
      int nrows = heights.size();
      int ncols = heights[0].size();
      dd = vector<vector<int>>(nrows, vector<int>(ncols, INT_MAX));
      dd[0][0] = 0;

      Q.push_back({0,0});

      vector<int> dx = {0, 0, 1, -1};
      vector<int> dy = {1, -1, 0, 0};
      set<pair<int, int>> S;// set of <r,c>'s
      while (!Q.empty()) {
        const auto [r,c] = Q[0];
        pop_heap(Q.begin(), Q.end(), [this] (const auto &lhs, const auto &rhs) { return dd[lhs.first][lhs.second] > dd[rhs.first][rhs.second]; });
        Q.pop_back();

        for (int i = 0; i < dx.size(); i++) {
          int r1 = r + dy[i];
          int c1 = c + dx[i];
          if (r1 >= 0 && r1 < nrows && c1 >=0 && c1 < ncols) {
            if (dd[r1][c1] > max(abs(heights[r1][c1] - heights[r][c]), dd[r][c])) {
              // relaxation
              dd[r1][c1] = max(abs(heights[r1][c1] - heights[r][c]), dd[r][c]);
              Q.push_back({r1,c1});
              push_heap(Q.begin(), Q.end(), [this] (const auto &lhs, const auto &rhs) { return dd[lhs.first][lhs.second] > dd[rhs.first][rhs.second]; });
            }
          }
        }
      }

      int ans = dd[nrows-1][ncols-1];
      return ans;
    }
};
