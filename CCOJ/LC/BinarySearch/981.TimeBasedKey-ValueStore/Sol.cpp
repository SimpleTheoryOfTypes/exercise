#include <vector>
#include <map>
#include <string>
using namespace std;

class TimeMap {
    map<string, vector<int>> mapIndex;
    map<string, vector<string>> mapStr;
public:
    TimeMap() {
    }

    void set(string key, string value, int timestamp) {
        if (mapIndex.find(key) == mapIndex.end()) {
            mapIndex[key] = {timestamp};
            mapStr[key] = {value};
        } else {
            mapIndex[key].push_back(timestamp);
            mapStr[key].push_back(value);
        }
    }

    string get(string key, int timestamp) {
      if (mapIndex.find(key) == mapIndex.end())
          return "";

      auto &nums = mapIndex[key];

      auto index = binarySearch(nums, timestamp);

      // Largest timestamp_prev < timestamp: Not Found
      if (index < 0)
          return "";
      return mapStr[key][index];
    }

    int binarySearch(vector<int> &nums, int timestamp) {
        int lo = 0;
        int hi = nums.size() - 1;

        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            if (timestamp == nums[mid])
                return mid;

            if (timestamp < nums[mid])
                hi = mid - 1;
            else
                lo = mid + 1;
        }

        return hi;
    }
};

/**
 * Your TimeMap object will be instantiated and called as such:
 * TimeMap* obj = new TimeMap();
 * obj->set(key,value,timestamp);
 * string param_2 = obj->get(key,timestamp);
 */
