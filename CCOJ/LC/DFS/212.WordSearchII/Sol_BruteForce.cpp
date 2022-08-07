#include <vector>
#include <map>
#include <string>
#include <iostream>
using namespace std;

class Solution {
  vector<int> directions = {-1, 1};
  bool Found = false;
  int nrows;
  int ncols;
public:
    void Constructor(const vector<vector<char>>& board) {
      nrows = board.size();
      ncols = board[0].size();
    }

    vector<string> findWords(const vector<vector<char>>& board, const vector<string>& words) {
      vector<string> ans;
      assert(!exist(board, "pea"));
      for (auto word : words) {
        // reset Found to false
        Found = false;
        if (exist(board, word)) {
          ans.push_back(word);
        }
      }
      return ans;
    }

    bool exist(const vector<vector<char>>& board, const string &word) {
      Constructor(board);

      // Initialize bitMask to a nrows x ncols grid of all 0's.
      // Create a vector containing "nrows" vectors each of size "ncols".
      vector<vector<int>> bitMask(nrows, vector<int>(ncols));
      for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
          bitMask[i][j] = 0;
        }
      }

      /* print2DVector(bitMask); */
      string partialResult;

      for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
          partialResult = board[i][j];
          reset2DVector(bitMask);
          bitMask[i][j] = 1;
          //print2DVector(bitMask);
          //std::cout << "(SimpleTheoryOfTypes) Staring from " << i << ", " << j << ": " << partialResult << std::endl;
          generate(word, board, partialResult, bitMask, i, j);
        }
      }

      return Found;
    }

    void print2DVector(const vector<vector<int>> &x) {
      for (auto v : x) {
        for (auto r : v) {
          std::cout << "" << r << ",";
        }
        std::cout << "\n";
      }
    }

    void reset2DVector(vector<vector<int>> &x) {
      for (auto &v : x) {
        for (auto &r : v) {
          r = 0;
        }
      }
    }

    void generate(const string &word, const vector<vector<char>>& board, string &partialResult, vector<vector<int>> &bitMask, int posI, int posJ) {
      if (partialResult[partialResult.size()-1] != word[partialResult.size()-1])
        return;

      if (partialResult == word) {
        std::cout << "(SimpleTheoryOfTypes) Matching word: " << partialResult << std::endl;
        Found = true;
      }

      // posI and posJ are the current positions.
      // bitMask encodes cells that have been traversed (0: not, 1: traversed).
      // partialResult collects partial string collected so far.
      // Move horizontally:
      for (auto i : directions) {
        auto nextI = posI + i;
        if (nextI < nrows && nextI >= 0) {
          if (bitMask[nextI][posJ] == 0) {
            partialResult.insert(partialResult.end(), board[nextI][posJ]);
            bitMask[nextI][posJ] = 1;
            generate(word, board, partialResult, bitMask, nextI, posJ);
            partialResult.erase(partialResult.end() - 1);
            bitMask[nextI][posJ] = 0;
          }
        }
      }

      // Move vertically:
      for (auto j : directions) {
        auto nextJ = posJ + j;
        if (nextJ < ncols && nextJ >= 0) {
          if (bitMask[posI][nextJ] == 0) {
            partialResult.insert(partialResult.end(), board[posI][nextJ]);
            bitMask[posI][nextJ] = 1;
            generate(word, board, partialResult, bitMask, posI, nextJ);
            partialResult.erase(partialResult.end() - 1);
            bitMask[posI][nextJ] = 0;
          }
        }
      }
    }
};

int main() {
  auto sol = Solution();
  /* auto ans = sol.exist({{'A','B','C','E'},{'S','F','C','S'},{'A','D','E','E'}}, "ABCCED"); */
  /* auto ans = sol.exist({{'A'}}, "A"); */
  //{{"L","L","A","B","L","D"},{"G","A","B","A","L","L"},{"A","B","C","B","J","A"},{"L","E","D","H","I","L"}} "GABALJIHDCBA"
  /* auto ans = sol.exist({{'L','L','A','B','L','D'},{'G','A','B','A','L','L'},{'A','B','C','B','J','A'},{'L','E','D','H','I','L'}}, "GABALJIHDCBA"); */
  /* auto ans = sol.exist({{'o','a','a','n'},{'e','t','a','e'},{'i','h','k','r'},{'i','f','l','v'}}, "pea"); */
  /* auto ans = sol.findWords({{'o','a','a','n'},{'e','t','a','e'},{'i','h','k','r'},{'i','f','l','v'}}, {"oath","pea","eat","rain"}); */
  auto ans = sol.findWords({{'b','a','b','a','b','a','b','a','b','a'},{'a','b','a','b','a','b','a','b','a','b'},{'b','a','b','a','b','a','b','a','b','a'},{'a','b','a','b','a','b','a','b','a','b'},{'b','a','b','a','b','a','b','a','b','a'},{'a','b','a','b','a','b','a','b','a','b'},{'b','a','b','a','b','a','b','a','b','a'},{'a','b','a','b','a','b','a','b','a','b'},{'b','a','b','a','b','a','b','a','b','a'},{'a','b','a','b','a','b','a','b','a','b'}}, {"ababababaa","ababababab","ababababac","ababababad","ababababae","ababababaf","ababababag","ababababah","ababababai","ababababaj","ababababak","ababababal","ababababam","ababababan","ababababao","ababababap","ababababaq","ababababar","ababababas","ababababat","ababababau","ababababav","ababababaw","ababababax","ababababay","ababababaz","ababababba","ababababbb","ababababbc","ababababbd","ababababbe","ababababbf","ababababbg","ababababbh","ababababbi","ababababbj","ababababbk","ababababbl","ababababbm","ababababbn","ababababbo","ababababbp","ababababbq","ababababbr","ababababbs","ababababbt","ababababbu","ababababbv","ababababbw","ababababbx","ababababby","ababababbz","ababababca","ababababcb","ababababcc","ababababcd","ababababce","ababababcf","ababababcg","ababababch","ababababci","ababababcj","ababababck","ababababcl","ababababcm","ababababcn","ababababco","ababababcp","ababababcq","ababababcr","ababababcs","ababababct","ababababcu","ababababcv","ababababcw","ababababcx","ababababcy","ababababcz","ababababda","ababababdb","ababababdc","ababababdd","ababababde","ababababdf","ababababdg","ababababdh","ababababdi","ababababdj","ababababdk","ababababdl","ababababdm","ababababdn","ababababdo","ababababdp","ababababdq","ababababdr","ababababds","ababababdt","ababababdu","ababababdv","ababababdw","ababababdx","ababababdy","ababababdz","ababababea","ababababeb","ababababec","ababababed","ababababee","ababababef","ababababeg","ababababeh","ababababei","ababababej","ababababek","ababababel","ababababem","ababababen","ababababeo","ababababep","ababababeq","ababababer","ababababes","ababababet","ababababeu","ababababev","ababababew","ababababex","ababababey","ababababez","ababababfa","ababababfb","ababababfc","ababababfd","ababababfe","ababababff","ababababfg","ababababfh","ababababfi","ababababfj","ababababfk","ababababfl","ababababfm","ababababfn","ababababfo","ababababfp","ababababfq","ababababfr","ababababfs","ababababft","ababababfu","ababababfv","ababababfw","ababababfx","ababababfy","ababababfz","ababababga","ababababgb","ababababgc","ababababgd","ababababge","ababababgf","ababababgg","ababababgh","ababababgi","ababababgj","ababababgk","ababababgl","ababababgm","ababababgn","ababababgo","ababababgp","ababababgq","ababababgr","ababababgs","ababababgt","ababababgu","ababababgv","ababababgw","ababababgx","ababababgy","ababababgz","ababababha","ababababhb","ababababhc","ababababhd","ababababhe","ababababhf","ababababhg","ababababhh","ababababhi","ababababhj","ababababhk","ababababhl","ababababhm","ababababhn","ababababho","ababababhp","ababababhq","ababababhr","ababababhs","ababababht","ababababhu","ababababhv","ababababhw","ababababhx","ababababhy","ababababhz","ababababia","ababababib","ababababic","ababababid","ababababie","ababababif","ababababig","ababababih","ababababii","ababababij","ababababik","ababababil","ababababim","ababababin","ababababio","ababababip","ababababiq","ababababir","ababababis","ababababit","ababababiu","ababababiv","ababababiw","ababababix","ababababiy","ababababiz","ababababja","ababababjb","ababababjc","ababababjd","ababababje","ababababjf","ababababjg","ababababjh","ababababji","ababababjj","ababababjk","ababababjl","ababababjm","ababababjn","ababababjo","ababababjp","ababababjq","ababababjr","ababababjs","ababababjt","ababababju","ababababjv","ababababjw","ababababjx","ababababjy","ababababjz","ababababka","ababababkb","ababababkc","ababababkd","ababababke","ababababkf","ababababkg","ababababkh","ababababki","ababababkj","ababababkk","ababababkl","ababababkm","ababababkn","ababababko","ababababkp","ababababkq","ababababkr","ababababks","ababababkt","ababababku","ababababkv","ababababkw","ababababkx","ababababky","ababababkz","ababababla","abababablb","abababablc","ababababld","abababable","abababablf","abababablg","abababablh","ababababli","abababablj","abababablk","ababababll","abababablm","ababababln","abababablo","abababablp","abababablq","abababablr","ababababls","abababablt","abababablu","abababablv","abababablw","abababablx","abababably","abababablz","ababababma","ababababmb","ababababmc","ababababmd","ababababme","ababababmf","ababababmg","ababababmh","ababababmi","ababababmj","ababababmk","ababababml","ababababmm","ababababmn","ababababmo","ababababmp","ababababmq","ababababmr","ababababms","ababababmt","ababababmu","ababababmv","ababababmw","ababababmx","ababababmy","ababababmz","ababababna","ababababnb","ababababnc","ababababnd","ababababne","ababababnf","ababababng","ababababnh","ababababni","ababababnj","ababababnk","ababababnl","ababababnm","ababababnn","ababababno","ababababnp","ababababnq","ababababnr","ababababns","ababababnt","ababababnu","ababababnv","ababababnw","ababababnx","ababababny","ababababnz","ababababoa","ababababob","ababababoc","ababababod","ababababoe","ababababof","ababababog","ababababoh","ababababoi","ababababoj","ababababok","ababababol","ababababom","ababababon","ababababoo","ababababop","ababababoq","ababababor","ababababos","ababababot","ababababou","ababababov","ababababow","ababababox","ababababoy","ababababoz","ababababpa","ababababpb","ababababpc","ababababpd","ababababpe","ababababpf","ababababpg","ababababph","ababababpi","ababababpj","ababababpk","ababababpl","ababababpm","ababababpn","ababababpo","ababababpp","ababababpq","ababababpr","ababababps","ababababpt","ababababpu","ababababpv","ababababpw","ababababpx","ababababpy","ababababpz","ababababqa","ababababqb","ababababqc","ababababqd","ababababqe","ababababqf","ababababqg","ababababqh","ababababqi","ababababqj","ababababqk","ababababql","ababababqm","ababababqn","ababababqo","ababababqp","ababababqq","ababababqr","ababababqs","ababababqt","ababababqu","ababababqv","ababababqw","ababababqx","ababababqy","ababababqz","ababababra","ababababrb","ababababrc","ababababrd","ababababre","ababababrf","ababababrg","ababababrh","ababababri","ababababrj","ababababrk","ababababrl","ababababrm","ababababrn","ababababro","ababababrp","ababababrq","ababababrr","ababababrs","ababababrt","ababababru","ababababrv","ababababrw","ababababrx","ababababry","ababababrz","ababababsa","ababababsb","ababababsc","ababababsd","ababababse","ababababsf","ababababsg","ababababsh","ababababsi","ababababsj","ababababsk","ababababsl","ababababsm","ababababsn","ababababso","ababababsp","ababababsq","ababababsr","ababababss","ababababst","ababababsu","ababababsv","ababababsw","ababababsx","ababababsy","ababababsz","ababababta","ababababtb","ababababtc","ababababtd","ababababte","ababababtf","ababababtg","ababababth","ababababti","ababababtj","ababababtk","ababababtl","ababababtm","ababababtn","ababababto","ababababtp","ababababtq","ababababtr","ababababts","ababababtt","ababababtu","ababababtv","ababababtw","ababababtx","ababababty","ababababtz","ababababua","ababababub","ababababuc","ababababud","ababababue","ababababuf","ababababug","ababababuh","ababababui","ababababuj","ababababuk","ababababul","ababababum","ababababun","ababababuo","ababababup","ababababuq","ababababur","ababababus","ababababut","ababababuu","ababababuv","ababababuw","ababababux","ababababuy","ababababuz","ababababva","ababababvb","ababababvc","ababababvd","ababababve","ababababvf","ababababvg","ababababvh","ababababvi","ababababvj","ababababvk","ababababvl","ababababvm","ababababvn","ababababvo","ababababvp","ababababvq","ababababvr","ababababvs","ababababvt","ababababvu","ababababvv","ababababvw","ababababvx","ababababvy","ababababvz","ababababwa","ababababwb","ababababwc","ababababwd","ababababwe","ababababwf","ababababwg","ababababwh","ababababwi","ababababwj","ababababwk","ababababwl","ababababwm","ababababwn","ababababwo","ababababwp","ababababwq","ababababwr","ababababws","ababababwt","ababababwu","ababababwv","ababababww","ababababwx","ababababwy","ababababwz","ababababxa","ababababxb","ababababxc","ababababxd","ababababxe","ababababxf","ababababxg","ababababxh","ababababxi","ababababxj","ababababxk","ababababxl","ababababxm","ababababxn","ababababxo","ababababxp","ababababxq","ababababxr","ababababxs","ababababxt","ababababxu","ababababxv","ababababxw","ababababxx","ababababxy","ababababxz","ababababya","ababababyb","ababababyc","ababababyd","ababababye","ababababyf","ababababyg","ababababyh","ababababyi","ababababyj","ababababyk","ababababyl","ababababym","ababababyn","ababababyo","ababababyp","ababababyq","ababababyr","ababababys","ababababyt","ababababyu","ababababyv","ababababyw","ababababyx","ababababyy","ababababyz","ababababza","ababababzb","ababababzc","ababababzd","ababababze","ababababzf","ababababzg","ababababzh","ababababzi","ababababzj","ababababzk","ababababzl","ababababzm","ababababzn","ababababzo","ababababzp","ababababzq","ababababzr","ababababzs","ababababzt","ababababzu","ababababzv","ababababzw","ababababzx","ababababzy","ababababzz"});
  for (auto a : ans)
    std::cout << "(SimpleTheoryOfTypes) " << a << std::endl;
  return 0;
}
