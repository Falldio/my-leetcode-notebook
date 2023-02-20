# Backtracking

## [Combination Sum](https://leetcode.com/problems/combination-sum)

A: 决策树，分叉点在于是否使用某一元素。

```cpp
class Solution {
public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {        
        vector<int> curr;
        vector<vector<int>> result;
        
        dfs(candidates, target, 0, 0, curr, result);
        return result;
    }
private:
    void dfs(vector<int>& candidates, int target, int sum, int start, vector<int>& curr, vector<vector<int>>& result) {
        if (sum > target) {
            return;
        }
        if (sum == target) {
            result.push_back(curr);
            return;
        }
        for (int i = start; i < candidates.size(); i++) {
            curr.push_back(candidates[i]);
            dfs(candidates, target, sum + candidates[i], i, curr, result);
            curr.pop_back();
        }
    }
};
```

```go
func combinationSum(candidates []int, target int) [][]int {
    ans := [][]int{}
    cur := []int{}
    for i := 0; i < len(candidates); i++ {
        cur = append(cur, candidates[i])
        // 可以重复使用
        dfs(i, target - candidates[i], candidates, &cur, &ans)
        cur = cur[:len(cur) - 1]
    }
    return ans
}

func dfs(idx, rest int, candidates []int, cur *[]int, ans *[][]int) {
    if rest == 0 {
        tmp := make([]int, len(*cur))
        copy(tmp, *cur)
        *ans = append(*ans, tmp)
        return
    }
    if rest < 0 {
        return
    }
    // idx开始，因为需要去重
    for i := idx; i < len(candidates); i++ {
        *cur = append(*cur, candidates[i])
        dfs(i, rest - candidates[i], candidates, cur, ans)
        *cur = (*cur)[:len(*cur) - 1]
    }
}
```

## [Word Search](https://leetcode.com/problems/word-search)

A: 回溯，向四个方向查找，标记查找过的单元格。

```cpp
class Solution {
public:
    bool exist(vector<vector<char>>& board, string word) {
        int m = board.size();
        int n = board[0].size();

        for (int row = 0; row < m; row++) {
            for (int col = 0; col < n; col++) {
                if(search(board, word, 0, m, n, row, col))  return true;
            }
        }
        return false;
    }

private:
    bool search(vector<vector<char>>& board, string word, int pos, int m, int n, int i, int j) {
        if (i < 0 || i >= m || j < 0 || j >= n || board[i][j] != word[pos] || board[i][j] == '!') {
            return false;
        }
        if (pos == word.size()-1) {
            return true;
        }

        board[i][j] = '!';
        if (search(board, word, pos+1, m, n, i+1, j)) return true;
        if (search(board, word, pos+1, m, n, i, j+1)) return true;
        if (search(board, word, pos+1, m, n, i-1, j)) return true;
        if (search(board, word, pos+1, m, n, i, j-1)) return true;
        board[i][j] = word[pos];
        return false;
    }
};
```

## [Generate Parentheses](https://leetcode.com/problems/generate-parentheses)

A: 标记当前左右括号数目。

```cpp
class Solution {
public:
    vector<string> generateParenthesis(int n) {
        vector<string> res;
        addingpar(res, "", n, 0);
        return res;
    }
    void addingpar(vector<string> &v, string str, int n, int m){
        if(n==0 && m==0) {
            v.push_back(str);
            return;
        }
        if(m > 0){ addingpar(v, str+")", n, m-1); }
        if(n > 0){ addingpar(v, str+"(", n-1, m+1); }
    }
};
```

## [Unique Paths III](https://leetcode.com/problems/unique-paths-iii)

A: 可优化：用`grid`存储是否访问的状态。

```cpp
class Solution {
public:
    int uniquePathsIII(vector<vector<int>>& grid) {
        int ans = 0;
        int m = grid.size();
        int n = grid[0].size();
        vector<vector<bool>> visited(m, vector<bool>(n, false));

        for (int i = 0; i < m; i ++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) {
                    dfs(ans, grid, visited, i, j, m, n);
                    return ans;
                }
            }
        }

        return ans;
    }

private:
    void dfs(int &ans, vector<vector<int>> &grid, vector<vector<bool>> &visited, int i, int j, int m, int n) {
        if (i < 0 || i >= m || j < 0 || j >= n || grid[i][j] == -1) {
            return;
        }

        if (grid[i][j] == 2) {
            for (int row = 0; row < m; row ++) {
                for (int col = 0; col < n; col++) {
                    if (grid[row][col] != -1 && grid[row][col] != 2) {
                        if (!visited[row][col]) return;
                    }
                }
            }
            ans++;
            return;
        }
        if (visited[i][j])  return;
        visited[i][j] = true;
        dfs(ans, grid, visited, i - 1, j, m, n);
        dfs(ans, grid, visited, i + 1, j, m, n);
        dfs(ans, grid, visited, i, j + 1, m, n);
        dfs(ans, grid, visited, i, j - 1, m, n);
        visited[i][j] = false;
    }
};
```

## [Subsets](https://leetcode.com/problems/subsets)

A: 遍历元素，有放入子集和不放两种情况。

```cpp
class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> ans;
        vector<int> set;
        backtracking(ans, set, nums, 0);
        return ans;
    }
private:
    void backtracking(vector<vector<int>> &ans, vector<int> &set, vector<int> &nums, int idx) {
        int len = nums.size();
        if (len == idx) {
            ans.push_back(set);
            return;
        }
        set.push_back(nums[idx]);
        backtracking(ans, set, nums, idx+1);
        set.pop_back();
        backtracking(ans, set, nums, idx+1);
    }
};
```

```go
func subsets(nums []int) [][]int {
    ans := [][]int{}
    cur := []int{}
    dfs(0, nums, &ans, &cur)
    return ans
}

func dfs(idx int, nums []int, ans *[][]int, cur *[]int) {
    if idx == len(nums) {
        tmp := make([]int, len(*cur))
        copy(tmp, *cur)
        *ans = append(*ans, tmp)
        return
    }
    dfs(idx + 1, nums, ans, cur)
    *cur = append(*cur, nums[idx])
    dfs(idx + 1, nums, ans, cur)
    *cur = (*cur)[:len(*cur) - 1]
}
```

## [Permutations](https://leetcode.com/problems/permutations)

A: 遍历并依次交换start和i处的元素，start == i时相当于不交换。

```cpp
class Solution {
public:
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> result;
        dfs(nums, 0, result);
        return result;
    }
private:
    void dfs(vector<int>& nums, int start, vector<vector<int>>& result) {
        if (start == nums.size()) {
            result.push_back(nums);
            return;
        }
        for (int i = start; i < nums.size(); i++) {
            swap(nums[i], nums[start]);
            dfs(nums, start + 1, result);
            swap(nums[i], nums[start]);
        }
    }
};
```

```go
var (
    ans [][]int
    cur []int
)
func permute(nums []int) [][]int {
    ans = [][]int{}
    cur = []int{}
    dfs(0, nums)
    return ans
}

func dfs(idx int, nums []int) {
    if idx == len(nums) {
        tmp := make([]int, len(nums))
        copy(tmp, nums)
        ans = append(ans, tmp)
        return
    }
    for i := idx; i < len(nums); i++ {
        nums[idx], nums[i] = nums[i], nums[idx]
        dfs(idx + 1, nums)
        nums[idx], nums[i] = nums[i], nums[idx]
    }
}
```

A: 每次取出一个元素，将其标记为visited，然后递归。

```go
var (
    ans [][]int
    cur []int
    visited map[int]struct{}
)
func permute(nums []int) [][]int {
    ans = [][]int{}
    cur = []int{}
    visited = map[int]struct{}{}
    dfs(nums)
    return ans
}

func dfs(nums []int) {
    if len(cur) == len(nums) {
        tmp := make([]int, len(nums))
        copy(tmp, cur)
        ans = append(ans, tmp)
    }
    for i := 0; i < len(nums); i++ {
        if _, ok := visited[nums[i]]; ok {
            continue
        }
        cur = append(cur, nums[i])
        visited[nums[i]] = struct{}{}
        dfs(nums)
        cur = cur[:len(cur) - 1]
        delete(visited, nums[i])
    }
}
```

## [Subsets II](https://leetcode.com/problems/subsets-ii)

A: 首先排序，然后跳过相同元素的情况。

```cpp
class Solution {
public:
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        
        vector<int> curr;
        vector<vector<int>> result;
        
        dfs(nums, 0, curr, result);
        return result;
    }
private:
    void dfs(vector<int>& nums, int start, vector<int>& curr, vector<vector<int>>& result) {
        result.push_back(curr);
        for (int i = start; i < nums.size(); i++) {
            if (i > start && nums[i] == nums[i - 1]) {
                continue;
            }
            curr.push_back(nums[i]);
            dfs(nums, i + 1, curr, result);
            curr.pop_back();
        }
    }
};
```

```go
func subsetsWithDup(nums []int) [][]int {
    ans := [][]int{}
    cur := []int{}
    sort.Ints(nums)
    dfs(0, nums, &cur, &ans)
    return ans
}

func dfs(idx int, nums []int, cur *[]int, ans *[][]int) {
    tmp := make([]int, len(*cur))
    copy(tmp, *cur)
    *ans = append(*ans, tmp)
    for i := idx; i < len(nums); i++ {
        if i > idx && nums[i] == nums[i - 1] {
            continue
        }
        *cur = append(*cur, nums[i])
        dfs(i+1, nums, cur, ans)
        *cur = (*cur)[:len(*cur) - 1]
    }
}
```

## [Combination Sum II](https://leetcode.com/problems/combination-sum-ii)

A: 排序后跳过重复元素（去重），相比于用map存储visited内存占用更小。

```cpp
class Solution {
public:
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        sort(candidates.begin(), candidates.end());
        vector<vector<int>> ans;
        vector<int> cur;
        dfs(ans, cur, candidates, target, 0);
        return ans;
    }
private:
    void dfs(vector<vector<int>> &ans, vector<int> &cur, vector<int> &candidates, int target, int start) {
        if (target == 0) {
            ans.push_back(cur);
            return;
        } else if (target < 0) {
            return;
        }
        for (int i = start; i < candidates.size(); i++) {
            if (i > start && candidates[i] == candidates[i-1])  continue;
            cur.push_back(candidates[i]);
            dfs(ans, cur, candidates, target - candidates[i], i + 1);
            cur.pop_back();
        }
    }
};
```

```go
func combinationSum2(candidates []int, target int) [][]int {
    ans := [][]int{}
    cur := []int{}
    sort.Ints(candidates)
    for i := 0; i < len(candidates); i++ {
        if i > 0 && candidates[i] == candidates[i - 1] {
            continue
        }
        cur = append(cur, candidates[i])
        dfs(i+1, target - candidates[i], candidates, &cur, &ans)
        cur = cur[:len(cur) - 1]
    }
    return ans
}

func dfs(idx, rest int, candidates []int, cur *[]int, ans *[][]int) {
    if rest == 0 {
        tmp := make([]int, len(*cur))
        copy(tmp, *cur)
        *ans = append(*ans, tmp)
        return
    }
    if rest < 0 {
        return
    }
    for i := idx; i < len(candidates); i++ {
        if i > idx && candidates[i] == candidates[i - 1] {
            continue
        }
        *cur = append(*cur, candidates[i])
        dfs(i+1, rest - candidates[i], candidates, cur, ans)
        *cur = (*cur)[:len(*cur) - 1]
    }
} 
```

## [Palindrome Partitioning](https://leetcode.com/problems/palindrome-partitioning)

A: 在可判断为回文处切分，然后回溯到未切分状态。

```cpp
class Solution {
public:
    vector<vector<string>> partition(string s) {
        vector<vector<string>> ans;
        vector<string> cur;
        dfs(ans, cur, s, 0);
        return ans;
    }
private:
    void dfs(vector<vector<string>> &ans, vector<string> &cur, string s, int start) {
        if (start == s.size()) {
            ans.push_back(cur);
            return;
        }
        for (int i = start; i < s.size(); i++) {
            if (isPalindrome(s, start, i)) {
                cur.push_back(s.substr(start, i - start + 1));
                dfs(ans, cur, s, i + 1);
                cur.pop_back();
            } else {
                continue;
            }
        }
    }
    bool isPalindrome(string s, int start, int end) {
        while (start <= end) {
            if (s[start++] != s[end--]) {
                return false;
            }
        }
        return true;
    }
};
```

```go
func partition(s string) [][]string {
    ans := [][]string{}
    cur := []string{}
    dfs(s, 0, &cur, &ans)
    return ans
}

func dfs(s string, idx int, cur *[]string, ans *[][]string) {
    if idx == len(s) {
        tmp := make([]string, len(*cur))
        copy(tmp, *cur)
        *ans = append(*ans, tmp)
        return
    }
    for i := idx + 1; i <= len(s); i++ {
        sub := s[idx:i]
        if !isValid(sub) {
            continue
        }
        *cur = append(*cur, sub)
        dfs(s, i, cur, ans)
        *cur = (*cur)[:len(*cur) - 1]
    }
}

func isValid(s string) bool {
    i, j := 0, len(s) - 1
    for i < j {
        if s[i] != s[j] {
            return false
        }
        i++
        j--
    }
    return true
}
```

## [Letter Combinations of a Phone Number](https://leetcode.com/problems/letter-combinations-of-a-phone-number)

A: 建立数字与字母映射关系。

```cpp
class Solution {
public:
    vector<string> letterCombinations(string digits) {
        if (digits.empty()) {
            return {};
        }
        
        unordered_map<char, string> m = {
            {'2', "abc"},
            {'3', "def"},
            {'4', "ghi"},
            {'5', "jkl"},
            {'6', "mno"},
            {'7', "pqrs"},
            {'8', "tuv"},
            {'9', "wxyz"}
        };
        string curr = "";
        vector<string> result;
        
        dfs(digits, 0, m, curr, result);
        return result;
    }
private:
    void dfs(string digits, int index, unordered_map<char, string>& m, string& curr, vector<string>& result) {
        if (index == digits.size()) {
            result.push_back(curr);
            return;
        }
        string str = m[digits[index]];
        for (int i = 0; i < str.size(); i++) {
            curr.push_back(str[i]);
            dfs(digits, index + 1, m, curr, result);
            curr.pop_back();
        }
    }
};
```

```go
var (
    m = map[byte][]rune {
        '2': {'a', 'b', 'c'},
        '3': {'d', 'e', 'f'},
        '4': {'g', 'h', 'i'},
        '5': {'j', 'k', 'l'},
        '6': {'m', 'n', 'o'},
        '7': {'p', 'q', 'r', 's'},
        '8': {'t', 'u', 'v'},
        '9': {'w', 'x', 'y', 'z'}}
)

func letterCombinations(digits string) []string {
    ans := []string{}
    cur := ""
    if len(digits) == 0 {
        return ans
    }
    dfs(0, digits, cur, &ans)
    return ans
}

func dfs(idx int, digits, cur string, ans *[]string) {
    if idx == len(digits) {
        *ans = append(*ans, cur)
        return
    }
    for _, r := range m[digits[idx]] {
        dfs(idx + 1, digits, cur + string(r), ans)
    }
}
```

## [N Queens](https://leetcode.com/problems/n-queens)

A: 逐行放皇后，检测是否可以放置。

```cpp
class Solution {
public:
    vector<vector<string>> ret;
    bool is_valid(vector<string> &board, int row, int col){
        // check col
        for (int i = row;i >= 0; --i)
            if (board[i][col] == 'Q') return false;
        // check left diagonal
        for (int i = row,j = col;i >= 0 &&j >= 0; --i, --j)
            if (board[i][j] == 'Q') return false;
        // check right diagonal
        for (int i = row, j = col; i >= 0 && j < board.size(); --i, ++j)
            if (board[i][j] == 'Q') return false;
        return true;
    }
    void dfs(vector<string> &board, int row){
        // exit condition
        if (row == board.size()){
            ret.push_back(board);
            return;
        }
        // iterate every possible position
        for (int i = 0; i < board.size(); ++i){
            if (is_valid(board, row, i)){
                // make decision
                board[row][i] = 'Q';
                // next iteration
                dfs(board, row+1);
                // back-tracking
                board[row][i] = '.';
            }
        }
    }
    vector<vector<string>> solveNQueens(int n) {
        // return empty if n <= 0
        if (n <= 0) return {{}};
        vector<string> board(n, string(n, '.'));
        dfs(board, 0);
        return ret;
    }
};
```

## [N-Queens II](https://leetcode.com/problems/n-queens-ii)

A: 逐行放皇后，区别在需要维护一个答案数字。

```cpp
class Solution {
public:
    int queen[9];
    bool check(int &r,int &c,int n){
        for(int i=0;i<r;i++){
            // here we have to check from before rows any queen(queen[i]) is placed to attk the cur level queen;
            int pre_row=i; // previous row
            int pre_col=queen[i]; // previous col is stored in queen[i]
            // checking for col collison as rows cant be && and for diagonal attk
            if(pre_col==c or abs(r-pre_row)==abs(c-pre_col)) return false;
        }
        return true;
    }
    int bt(int level,int n){
        // base conditon
        
        if(level==n) return 1;
        // return 1 as u made a board and placing queens from 0 to n-1 so u came out of board
        
        int ans=0;
        // exploring choices and computation
        for(int col=0;col<n;col++){
            if(check(level,col,n)){
                // check
                queen[level]=col;
                // move
                ans+=bt(level+1,n);
                queen[level]=-1;
            }
        }
        // return count of ways to place queen from this row to n-1/ last row
        return ans;
    }
    int totalNQueens(int n) {
        memset(queen,-1,sizeof(queen));
        return bt(0,n);
    }
};
```

## [Combinations](https://leetcode.com/problems/combinations)

A: backtracking + DFS。

```cpp
class Solution {
public:
    vector<vector<int>> combine(int n, int k) {
        vector<vector<int>> ans;
        vector<int> cur;
        for (int i = 1; i <= n; i++) {
            cur.push_back(i);
            dfs(ans, cur, i, n, k);
            cur.pop_back();
        }
        return ans;
    }
private:
    void dfs(vector<vector<int>> &ans, vector<int> &cur, int end, int n, int k) {
        if (cur.size() > k) return;
        if (cur.size() == k) {
            ans.push_back(cur);
            return;
        }
        for (int i = end + 1; i <= n; i++) {
            cur.push_back(i);
            dfs(ans, cur, i, n, k);
            cur.pop_back();
        }
    }
};
```

```go
func combine(n int, k int) [][]int {
    ans := [][]int{}
    cur := []int{}

    for i := 1; i <= n; i++ {
        cur = append(cur, i)
        dfs(i + 1, n, k, &cur, &ans)
        cur = cur[:len(cur) - 1]
    }
    return ans
}

func dfs(num, n, k int, cur *[]int, ans *[][]int) {
    if len(*cur) == k {
        tmp := make([]int, k)
        copy(tmp, *cur)
        *ans = append(*ans, tmp)
        return
    }
    if num > n {
        return
    }
    for i := num; i <= n; i++ {
        *cur = append(*cur, i)
        dfs(i + 1, n, k, cur, ans)
        *cur = (*cur)[:len(*cur) - 1]
    }
}
```

## [Permutations II](https://leetcode.com/problems/permutations-ii)

A: DFS+回溯。

```cpp
class Solution {
public:
    void helper(vector<vector<int>>& res, vector<int>& nums, int pos) {
        
        if (pos == nums.size()) {
            res.push_back(nums);
        } else {
            for (int i = pos; i < nums.size(); ++i) {
                if (i > pos && nums[i] == nums[pos]) continue;
                swap(nums[pos], nums[i]);
                helper(res, nums, pos + 1);

            }
            // restore nums
            for (int i = nums.size() - 1; i > pos; --i) {
                swap(nums[pos], nums[i]);
            }
        }
    }

    vector<vector<int>> permuteUnique(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        vector<vector<int>> res;
        helper(res, nums, 0);
        return res;
    }
};
```

A: 依次取出元素进行排列，注意去重。

```go
var (
    res [][]int
    path  []int
    st    []bool   // state的缩写
)
func permuteUnique(nums []int) [][]int {
    res, path = make([][]int, 0), make([]int, 0, len(nums))
    st = make([]bool, len(nums))
    sort.Ints(nums)
    dfs(nums, 0)
    return res
}

func dfs(nums []int, cur int) {
    if cur == len(nums) {
        tmp := make([]int, len(path))
        copy(tmp, path)
        res = append(res, tmp)
    }
    for i := 0; i < len(nums); i++ {
        if i != 0 && nums[i] == nums[i-1] && !st[i-1] {  // 去重，用st来判别是深度还是广度
            continue
        }
        if !st[i] {
            path = append(path, nums[i])
            st[i] = true
            dfs(nums, cur + 1)
            st[i] = false
            path = path[:len(path)-1]
        }
    }
}
```

## [Restore Ip Addresses](https://leetcode.com/problems/restore-ip-addresses)

A: Backtracking + DFS。

```go
func restoreIpAddresses(s string) []string {
    var res []string

    var backtrack func(int, int, string)
    backtrack = func(left, numberOfDots int, currentCombination string) {
        if left > len(s)-1 {
            return
        }

        if numberOfDots == 1 {
            if s[left] == '0' && len(s)-left > 1 {
                return
            }

            if isValidIpAddress(s[left:]) {
                res = append(res, currentCombination+s[left:])
            }

            return
        }

        if s[left] == '0' {
            backtrack(left+1, numberOfDots-1, currentCombination+s[left:left+1]+".")
        } else {
            for i := left + 1; i < len(s); i++ {
                if isValidIpAddress(s[left:i]) {
                    backtrack(i, numberOfDots-1, currentCombination+s[left:i]+".")
                }
            }
        }
    }

    backtrack(0, 4, "")

    return res
}

func isValidIpAddress(str string) bool {
    val, err := strconv.Atoi(str)

    if err != nil {
        panic(err)
    }

    return val >= 0 && val <= 255
}
```

## [Matchsticks to Square](https://leetcode.com/problems/matchsticks-to-square)

A: DFS + backtracking。

```cpp
class Solution {
    bool dfs(vector<int> &sidesLength,const vector<int> &matches, int index, const int target) {
        if (index == matches.size())
            return sidesLength[0] == sidesLength[1] && sidesLength[1] == sidesLength[2] && sidesLength[2] == sidesLength[3];
        for (int i = 0; i < 4; ++i) {
            if (sidesLength[i] + matches[index] > target) // 检查是否已经超过理想边长
                continue;
            int j = i;
            while (--j >= 0) // 如果已存在相等边长，则说明这种情况之前已经考虑过
                if (sidesLength[i] == sidesLength[j]) 
                    break;
            if (j != -1) continue;
            sidesLength[i] += matches[index];
            if (dfs(sidesLength, matches, index + 1, target))
                return true;
            sidesLength[i] -= matches[index];
        }
        return false;
    }
public:
    bool makesquare(vector<int>& nums) {
        if (nums.size() < 4) return false;
        int sum = 0;
        for (const int val: nums) {
            sum += val;
        }
        if (sum % 4 != 0) return false;
        sort(nums.begin(), nums.end(), [](const int &l, const int &r){return l > r;}); // 优先分配最长边长，这样更容易达到true case
        vector<int> sidesLength(4, 0);
        return dfs(sidesLength, nums, 0, sum / 4);
    }
};
```

## [Non-decreasing Subsequences](https://leetcode.com/problems/non-decreasing-subsequences)

A: 题目条件中的数组不能排序，因此只能用set避免元素重复。

```cpp
class Solution {
public:
    vector<vector<int>> findSubsequences(vector<int>& nums) {
        vector<vector<int>> res;
        vector<int> seq;
        dfs(res, seq, nums, 0);
        return res;
    }
    
    void dfs(vector<vector<int>>& res, vector<int>& seq, vector<int>& nums, int pos) {
        if(seq.size() > 1) res.push_back(seq);
        unordered_set<int> hash;
        for(int i = pos; i < nums.size(); ++i) {
            if((seq.empty() || nums[i] >= seq.back()) && hash.find(nums[i]) == hash.end()) {
                seq.push_back(nums[i]);
                dfs(res, seq, nums, i + 1);
                seq.pop_back();
                hash.insert(nums[i]);
            }
        }
    }
};
```

```go
func findSubsequences(nums []int) [][]int {
    ans := [][]int{}
    cur := []int{}
    dfs(0, nums, &ans, &cur)
    return ans
}

func dfs(idx int, nums []int, ans *[][]int, cur *[]int) {
    if len(*cur) >= 2 {
        tmp := make([]int, len(*cur))
        copy(tmp, *cur)
        *ans = append(*ans, tmp)
    }
    m := map[int]struct{}{}
    for i := idx; i < len(nums); i++ {
        _, ok := m[nums[i]]
        if (len(*cur) > 0 && nums[i] < (*cur)[len(*cur) - 1]) || ok {
            continue
        }
        m[nums[i]] = struct{}{}
        *cur = append(*cur, nums[i])
        dfs(i + 1, nums, ans, cur)
        *cur = (*cur)[:len(*cur) - 1]
    }
}
```

## [Splitting a String Into Descending Consecutive Values](https://leetcode.com/problems/splitting-a-string-into-descending-consecutive-values)

A: 依次切分字符串，进行条件检查。

```cpp
long long MX = 999999999999;
class Solution {
public:
    bool dfs(string &s, long long prev, int idx, int cnt) {
        if(idx == s.size() ) return cnt > 1;
        long long num = 0;
        for(int i = idx; i < s.size(); i++) {
            num = num *  10l + s[i] - '0';
            if(num > MX) break;
            if(num == prev - 1 || prev == -1) {
                if(dfs(s, num, i + 1, cnt + 1)) return true;
            }
            if(num > prev && prev != -1) break;
        }
        return false;
    }
    
    bool splitString(string s) {
        if(s.size() <= 1) return false;
        return dfs(s, -1, 0, 0);
    }
};
```

## [Find Unique Binary String](https://leetcode.com/problems/find-unique-binary-string)

A: 字符串长度与要比较的数组长度相等，可保证数组中字符串至少有1位与答案不同。

[Cantor's Diagonalization](https://en.wikipedia.org/wiki/Cantor%27s_diagonal_argument)

```cpp
class Solution {
public:
    string findDifferentBinaryString(vector<string>& nums) {
        string ans="";
        for(int i=0; i<nums.size(); i++) 
            ans+= nums[i][i]=='0' ? '1' : '0';          // Using ternary operator
        return ans;
    }
};
```

## [Maximum Length of a Concatenated String with Unique Characters](https://leetcode.com/problems/maximum-length-of-a-concatenated-string-with-unique-characters)

A: 0-1背包问题，DFS+回溯。

```cpp
class Solution {
public:
    int maxLength(vector<string>& arr, string str = "", int index = 0) {
        //Use set to check if the string contains all unique characters
        unordered_set<char>s(str.begin(), str.end());
        if (s.size() != ((int)str.length()))  // s存在重复字母
            return 0;
        
        int ret = str.length();
        for (int i = index; i < arr.size(); i++) // 从第i位开始考虑后续连接
            ret = max(ret, maxLength(arr, str+arr[i], i+1));

        return ret;
    }
};
```

A: DP。

`bitset`: 相当于二进制数组，支持位运算。

```cpp
class Solution {
public:
    int maxLength(vector<string>& A) {
        vector<bitset<26>> dp = {bitset<26>()};
        int res = 0;
        for (auto& s : A) {
            bitset<26> a;
            for (char c : s)
                a.set(c - 'a');
            int n = a.count();
            if (n < s.size()) continue; // 检查s中是否含有重复字母

            for (int i = dp.size() - 1; i >= 0; --i) {
                bitset c = dp[i];
                if ((c & a).any()) continue; // 存在重复字母
                dp.push_back(c | a);
                res = max(res, (int)c.count() + n);
            }
        }
        return res;
    }
};
```

## [Partition to K Equal Sum Subsets](https://leetcode.com/problems/partition-to-k-equal-sum-subsets)

A: DP + 位运算。

[详解](https://leetcode.com/problems/partition-to-k-equal-sum-subsets/solutions/480707/c-dp-bit-manipulation-in-20-lines/?orderBy=most_votes)

```cpp
class Solution {
public:
    int dp[(1<<16)+2]; // 根据题目条件，k最大为16
    bool canPartitionKSubsets(vector<int>& nums, int k) {
        int n = nums.size(), sum = 0;
        fill(dp, dp+(1<<16)+2, -1);
        dp[0] = 0;
        for (int i = 0; i < n; i++) sum += nums[i];
        if (sum % k) return false;
        int tar = sum/k;
        
        for (int mask = 0; mask < (1<<n); mask++) {
            if (dp[mask] == -1) continue;  // if current state is illegal, simply ignore it
            for (int i = 0; i < n; i++) {
                if (!(mask&(1<<i)) && dp[mask]+nums[i] <= tar) {  // if nums[i] is unchosen && choose nums[i] would not cross the target
                    dp[mask|(1<<i)] = (dp[mask]+nums[i]) % tar;
                }
            }
        }
        return dp[(1<<n)-1] == 0;
    }
};
```

## [Combination Sum III](https://leetcode.com/problems/combination-sum-iii)

A: 组合问题，回溯。

```go
func combinationSum3(k int, n int) [][]int {
    ans := [][]int{}
    cur := make([]int, 0, k)
    for i := 1; i <= 9 - k + 1; i++ {
        cur = append(cur, i)
        dfs(i + 1, k, n - i, &ans, &cur)
        cur = cur[:len(cur) - 1]
    }
    return ans
}

func dfs(num, k, rest int, ans *[][]int, cur *[]int) {
    if len(*cur) == k && rest == 0 {
        tmp := make([]int, k)
        copy(tmp, *cur)
        *ans = append(*ans, tmp)
        return
    }
    if rest < 0 || len(*cur) > k {
        return
    }

    for i := num; i <= 9; i++ {
        *cur = append(*cur, i)
        dfs(i + 1, k, rest - i, ans, cur)
        *cur = (*cur)[:len(*cur) - 1]
    }
}
```
