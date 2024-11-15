class Solution {
    const double PI = 3.14159265358979323846;
public:
    int visiblePoints(vector<vector<int>>& points, int angle, vector<int>& location) {
      vector<double> angles;
      int ansConst = 0;// points that are the same as location, should always be included in the answer.
      for (const auto &p : points) {
        int x = p[0];
        int y = p[1];
        if (x == location[0] && y == location[1]) {
          ansConst += 1;
          continue;
        }
          
        double angle_in_degrees = atan2(y - location[1], x - location[0]) * 180 / PI;
        angles.push_back(angle_in_degrees);
      }
      
      // simple case
      if (angles.empty())
        return ansConst;
      
      sort(angles.begin(), angles.end());
      vector<double> angles2;
      angles2.insert(angles2.end(), angles.begin(), angles.end());
      for (const auto &x : angles)
        angles2.push_back(x + 360);
      
      int end = -1; // such that angles[end] < start + angle
      for (int i = 0; i < angles.size(); i++) {
        if (angles[0] + angle >= angles[i])
          end = i;
      }
      assert(end >= 0);
      end = end + 1;
      // (invariant): [start = 0, end) is the maximally valid range starting from the first data point.
      int ans = end;
      for (int i = 1; i < angles.size(); i++) {
        int newEnd = end + 1;
        while (angles[i] + angle >= angles2[newEnd])// FIXMED
          newEnd++;
        end = newEnd; 
        // (invariant) [start = i, end) is the maximally valid range starting from the i-th data point.
        ans = max(end - i, ans);
      }
      
      return ans + ansConst;
    }
}
