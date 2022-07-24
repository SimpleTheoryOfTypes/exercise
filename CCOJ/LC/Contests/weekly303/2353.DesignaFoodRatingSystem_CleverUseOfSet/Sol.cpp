class FoodRatings {
    map<string, set<pair<int, string>>> rmp;// rating map: cusine -> <food, rating>
    map<string, string> f2c;// food to cusine
    map<string, int> f2r;// food to rating
public:
    FoodRatings(vector<string>& foods, vector<string>& cuisines, vector<int>& ratings) {
      for (int i = 0; i < foods.size(); i++) {
        auto &cs = cuisines[i];
        rmp[cs].insert(make_pair(-ratings[i], foods[i]));
        f2c[foods[i]] = cs; 
        f2r[foods[i]] = ratings[i]; 
      }
    }
    
    void changeRating(string food, int newRating) {
      const auto &cs = f2c[food];
      rmp[cs].erase( {-f2r[food], food});
      rmp[cs].insert({-newRating, food});
      f2r[food] = newRating;
    }
    
    string highestRated(string cuisine) {
       auto &rv = rmp[cuisine];
       return rv.begin()->second; 
    }
};


// My original solution using make_heap (correct, but TLE).
class FoodRatings {
    map<string, vector<pair<string,int>>> rmp;// rating map: cusine -> <food, rating>
    map<string, string> f2c;// food to cusine
    map<string, int> f2i; // food to it's index in rmp[its cuisine]
public:
    FoodRatings(vector<string>& foods, vector<string>& cuisines, vector<int>& ratings) {
      map<string, int> cusineIndices;
      for (int i = 0; i < foods.size(); i++) {
        auto &cs = cuisines[i];
        rmp[cs].push_back(make_pair(foods[i], ratings[i]));
        f2c[foods[i]] = cs; 
        f2i[foods[i]] = cusineIndices[cs];
        cusineIndices[cs] += 1;
      }
    }
    
    void changeRating(string food, int newRating) {
      const auto &cs = f2c[food];
      rmp[cs][f2i[food]].second = newRating; 
      //for (auto & pr : rmp[cs]) {
      //  if (pr.first == food) {
      //    pr.second = newRating;
      //    break;
      //  }
      //}
    }
    
    string highestRated(string cuisine) {
       auto rv = rmp[cuisine];
       make_heap(rv.begin(), rv.end(), [] (const auto &lhs, const auto &rhs) {
         return lhs.second < rhs.second || ((lhs.second == rhs.second) && (lhs.first > rhs.first));
       });      
       return rv[0].first; 
    }
};

/**
 * Your FoodRatings object will be instantiated and called as such:
 * FoodRatings* obj = new FoodRatings(foods, cuisines, ratings);
 * obj->changeRating(food,newRating);
 * string param_2 = obj->highestRated(cuisine);
 */

