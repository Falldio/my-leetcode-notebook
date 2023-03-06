# 1-D Dynamic Programming

## [Climbing Stairs](https://leetcode.com/problems/climbing-stairs)

A: 斐波那契数列。

```cpp
class Solution {
public:
    int climbStairs(int n) {
        if (n==0)   return 0;
        if (n==1)   return 1;
        if (n==2)   return 2;
        vector<int> ans(n, 0);
        ans[0] = 1;
        ans[1] = 2;

        for (int i = 2; i < n; i++) {
            ans[i] = ans[i-1] + ans[i-2];
        }

        return ans[n-1];
    }
};
```

```go
func climbStairs(n int) int {
    if n == 1 {
        return 1
    }
    if n == 2 {
        return 2
    }
    dp := make([]int, n + 1)
    dp[1] = 1
    dp[2] = 2
    for i := 3; i <= n; i++ {
        dp[i] = dp[i - 1] + dp[i - 2]
    }
    return dp[n]
}
```

## [House Robber](https://leetcode.com/problems/house-robber)

A: 当前house的最大值取决于前两个house的最大值，只用两个局部变量存储该值。

```cpp
class Solution {
public:
    int rob(vector<int>& nums) {
        int prev = 0;
        int curr = 0;
        int next = 0;
        
        for (int i = 0; i < nums.size(); i++) {
            next = max(prev + nums[i], curr);
            prev = curr;
            curr = next;
        }
        
        return curr;
    }
};
```

## [House Robber II](https://leetcode.com/problems/house-robber-ii)

A: 分解为两个House Robber，[0, n-2]和[1, n-1]。

```cpp
class Solution {
public:
    int rob(vector<int>& nums) {
        int n = nums.size(); 
        if (n < 2) return n ? nums[0] : 0;
        return max(robber(nums, 0, n - 2), robber(nums, 1, n - 1));
    }
private:
    int robber(vector<int>& nums, int l, int r) {
        int pre = 0, cur = 0;
        for (int i = l; i <= r; i++) {
            int temp = max(pre + nums[i], cur);
            pre = cur;
            cur = temp;
        }
        return cur;
    }
};
```

## [Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring)

A: 对每一个字符，左右滑动指针，进行奇数偶数两种情况判断，子串为回文是外串为回文的必要条件。

```cpp
class Solution {
public:
    string longestPalindrome(string s) {
        int maxStart = 0;
        int maxLength = 1;
        
        for (int i = 0; i < s.size() - 1; i++) {
            middleOut(s, i, i, maxStart, maxLength);
            middleOut(s, i, i + 1, maxStart, maxLength);
        }
        
        return s.substr(maxStart, maxLength);
    }
private:
    void middleOut(string s, int i, int j, int& maxStart, int& maxLength) {
        while (i >= 0 && j <= s.size() - 1 && s[i] == s[j]) {
            i--;
            j++;
        }
        if (j - i - 1 > maxLength) {
            maxStart = i + 1;
            maxLength = j - i - 1;
        }
    }
};
```

## [Palindromic Substrings](https://leetcode.com/problems/palindromic-substrings)

A: 对每一个字符，左右滑动指针，进行奇数偶数两种情况判断，子串为回文是外串为回文的必要条件。

```cpp
class Solution {
public:
    int countSubstrings(string s) {
        int ans = 0;
        for (int i = 0; i < s.size(); i++) {
            ans++;
            isPalindromic(ans, s, i, true);
            isPalindromic(ans, s, i, false);
        }

        return ans;
    }

private:
    void isPalindromic(int &ans, string s, int i, bool isEven) {
        int left, right;
        if (isEven) {
            left = i - 1;
            right = i + 1;
        } else {
            left = i;
            right = i + 1;
        }
        while (left >= 0 && right <= s.size()-1 && s[left] == s[right]) {
            ans++;
            left--;
            right++;
        }
    }
};
```

## [Decode Ways](https://leetcode.com/problems/decode-ways)

A: 长度为n的字符串解码的子问题为长度为n-1和长度为n-2的子问题解码，具体看前面能否出现两位数。

```cpp
class Solution {
public:
    int numDecodings(string s) {
        if (s[0] == '0') {
            return 0;
        }
        
        int n = s.size();
        
        vector<int> dp(n + 1);
        dp[0] = 1; // 空字符串
        dp[1] = 1; // 单字母
        
        for (int i = 2; i <= n; i++) {
            int ones = stoi(s.substr(i - 1, 1));
            if (ones >= 1 && ones <= 9) {
                dp[i] += dp[i - 1];
            }
            int tens = stoi(s.substr(i - 2, 2));
            if (tens >= 10 && tens <= 26) {
                dp[i] += dp[i - 2]; // 此时前面取出两位数的情况数目等同于单独取出i-2的次数（即i-2、i-1视为一个整体）
            }
        }
        
        return dp[n];
    }
};
```

## [Coin Change](https://leetcode.com/problems/coin-change)

A: 集齐n块钱的方法取决于集齐n-c块钱的方法，c为要使用的硬币面值。

[详解](https://leetcode.com/problems/coin-change/solutions/778548/c-dp-solution-explained-100-time-100-space/?orderBy=most_votes)

```cpp
class Solution {
public:
    int coinChange(vector<int>& coins, int n) {
        // creating the base dp array, with first value set to 0
        int dp[++n];
        dp[0] = 0;
        // more convenient to have the coins sorted
        sort(begin(coins), end(coins));
        // populating our dp array
        for (int i = 1; i < n; i++) {
            // setting dp[0] base value to 1, 0 for all the rest
            dp[i] = INT_MAX;
            for (int c: coins) {
                if (i - c < 0) break;
                // if it was a previously not reached cell, we do not add use it
                if (dp[i - c] != INT_MAX) dp[i] = min(dp[i], 1 + dp[i - c]);
            }
        }
        return dp[--n] == INT_MAX ? -1 : dp[n];
    }
};
```

## [Word Break](https://leetcode.com/problems/word-break)

A: 逐字母向后处理，逐字母向前回溯。

```cpp
class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        int len = s.size();
        vector<bool> dp(len + 1, false);
        unordered_set<string> dict;
        for (auto word : wordDict) {
            dict.insert(word);
        }
        dp[0] = true;

        for (int i = 1; i <= len; i++) {
            for (int j = i - 1; j >= 0; j--) {
                if (dp[j]) {
                    string word = s.substr(j, i - j);
                    if (dict.find(word) != dict.end()) {
                        dp[i] = true;
                        break;
                    }
                }
            }
        }

        return dp[len];
    }
};
```

## [Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence)

A: 遍历的同时存储当前位置的LIS，以供后续遍历回溯。

```cpp
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int len = nums.size();
        vector<int> lens(len + 1, INT_MIN);
        lens[0] = 0;
        int ans = 0;

        for (int i = 1; i < len + 1; i++) {
            for (int j = i - 1; j >= 0; j--) {
                if (lens[j] != INT_MIN) {
                    if (j == 0 || nums[i-1] > nums[j-1]) {
                        lens[i] = max(lens[j] + 1, lens[i]);
                        ans = max(lens[i], ans);
                    }
                }
            }
        }

        return ans;
    }
};
```

## [Min Cost Climbing Stairs](https://leetcode.com/problems/min-cost-climbing-stairs)

A: 两个变量分别存放前一步和前两步的代价。

```cpp
class Solution {
public:
    int minCostClimbingStairs(vector<int>& cost) {
        int downOne = 0;
        int downTwo = 0;
        
        for (int i = 2; i <= cost.size(); i++) {
            int temp = downOne;
            downOne = min(downOne + cost[i - 1], downTwo + cost[i - 2]);
            downTwo = temp;
        }
        
        return downOne;
    }
};
```

```go
func minCostClimbingStairs(cost []int) int {
    first, second := 0, 0
    for i := 2; i < len(cost) + 1; i++ {
        tmp := first
        first = min(first + cost[i - 1], second + cost[i - 2])
        second = tmp
    }
    return first
}

func min(i, j int) int {
    if i < j {
        return i
    } else {
        return j
    }
}
```

## [Partition Equal Subset Sum](https://leetcode.com/problems/partition-equal-subset-sum)

A: 用set存放所有中间结果。

![详解](https://leetcode.com/problems/partition-equal-subset-sum/solutions/1624939/c-python-5-simple-solutions-w-explanation-optimization-from-brute-force-to-dp-to-bitmask/?orderBy=most_votes)

```cpp
class Solution {
public:
    bool canPartition(vector<int>& nums) {
        int target = 0;
        for (int i = 0; i < nums.size(); i++) {
            target += nums[i];
        }
        if (target % 2 != 0) {
            return false;
        }
        target /= 2;
        
        unordered_set<int> dp; // 保存所有中间的‘和’结果
        dp.insert(0);
        
        for (int i = 0; i < nums.size(); i++) {
            unordered_set<int> dpNext;
            for (auto it = dp.begin(); it != dp.end(); it++) {
                if (*it + nums[i] == target) {
                    return true;
                }
                dpNext.insert(*it + nums[i]); // 选择nums[i]
                dpNext.insert(*it); // 舍弃nums[i]
            }
            dp = dpNext;
        }
        
        return false;
    }
};
```

## [Triangle](https://leetcode.com/problems/triangle)

A: 每一层的最小路径为本层元素加上子层最小路径。

```cpp
class Solution {
public:
    int minimumTotal(vector<vector<int>>& triangle) {
        int len = triangle.size();
        if (len == 1) return triangle[0][0];
        for (int row = len - 2; row >= 0; row--) {
            for (int i = 0; i < triangle[row].size(); i++) {
                triangle[row][i] += min(triangle[row+1][i], triangle[row+1][i+1]);
            }
        }
        return triangle[0][0];
    }
};
```

## [Delete And Earn](https://leetcode.com/problems/delete-and-earn)

A: take[i] = skip[i - 1] + values[i]。

```cpp
class Solution {
public:
    int deleteAndEarn(vector<int>& nums) {
        int n = 10001;
        vector<int> values(n, 0);
        for (int num : nums)
            values[num] += num;

        int take = 0, skip = 0;
        for (int i = 0; i < n; i++) {
            int takei = skip + values[i];
            int skipi = max(skip, take);
            take = takei;
            skip = skipi;
        }
        return max(take, skip);
    }
};
```

## [Flip String to Monotone Increasing](https://leetcode.com/problems/flip-string-to-monotone-increasing)

A: 子问题为前缀子串升序的最小代价。

```cpp
class Solution {
public:
    int minFlipsMonoIncr(const std::string& S, int counter_one  = 0, int counter_flip = 0) {
        for (auto ch : S) {
            if (ch == '1') {
                ++counter_one;
            } else {
                ++counter_flip;
            }
            counter_flip = std::min(counter_one, counter_flip);
        }
        return counter_flip;
    }
};
```

```go
func minFlipsMonoIncr(s string) int {
    ans := 0
    ones := 0
    for _, c := range s {
        if c == '1' {
            ones++
        } else {
            ans++
        }
        if ones < ans {
            ans = ones
        }
    }
    return ans
}
```

## [Paint House](https://www.lintcode.com/problem/515/description)

A: 记录rgb三色的历史费用，可进一步优化内存。

```cpp
class Solution {
public:
    /**
     * @param costs: n x 3 cost matrix
     * @return: An integer, the minimum cost to paint all houses
     */
    int minCost(vector<vector<int>> &costs) {
        int len = costs.size();
        if (len == 1) {
            return min({costs[0][0], costs[0][1], costs[0][2]});
        } else if (len == 0) {
            return 0;
        }
        vector<int> r(len), g(len), b(len);
        r[0] = costs[0][0];
        g[0] = costs[0][1];
        b[0] = costs[0][2];
        for (int i = 1; i < costs.size(); i++) {
            r[i] = costs[i][0] + min(g[i - 1], b[i - 1]);
            g[i] = costs[i][1] + min(r[i - 1], b[i - 1]);
            b[i] = costs[i][2] + min(r[i - 1], g[i - 1]);
        }
        return min({r[len - 1], g[len - 1], b[len - 1]});
    }
};
```

## [Combination Sum IV](https://leetcode.com/problems/combination-sum-iv)

A: 组合背包问题。

```cpp
class Solution {
public:
    int combinationSum4(vector<int>& nums, int target) {
        sort(nums.begin(), nums.end());
        vector<unsigned int> dp(1001, 0);
        dp[0] = 1;
        for (int i = 1; i <= target; i++) {
            for (int j = 0; j < nums.size(); j++) {
                if (i - nums[j] >= 0) {
                    dp[i] += dp[i - nums[j]];
                }
            }
        }
        return dp[target];
    }
};
```

## [Perfect Squares](https://leetcode.com/problems/perfect-squares)

A: DP。

```cpp
class Solution 
{
public:
    int numSquares(int n) 
    {
        if (n <= 0)
        {
            return 0;
        }
        
        // cntPerfectSquares[i] = the least number of perfect square numbers 
        // which sum to i. Note that cntPerfectSquares[0] is 0.
        vector<int> cntPerfectSquares(n + 1, INT_MAX);
        cntPerfectSquares[0] = 0;
        for (int i = 1; i <= n; i++)
        {
            // For each i, it must be the sum of some number (i - j*j) and 
            // a perfect square number (j*j).
            for (int j = 1; j*j <= i; j++)
            {
                cntPerfectSquares[i] = 
                    min(cntPerfectSquares[i], cntPerfectSquares[i - j*j] + 1);
            }
        }
        
        return cntPerfectSquares.back();
    }
};
```

## [Maximum Subarray Min-Product](https://leetcode.com/problems/maximum-subarray-min-product)

A: 前缀和记录数组前n个元素之和，monostack（升序堆）用于得到当前最小元素及subarray区间。

```cpp
class Solution {
public:
    int maxSumMinProduct(vector<int>& n) {
        long res = 0;
        vector<long> dp(n.size() + 1), st;
        for (int i = 0; i < n.size(); ++i)
           dp[i + 1] = dp[i] + n[i];
        for (int i = 0; i <= n.size(); ++i) {
            while (!st.empty() && (i == n.size() || n[st.back()] > n[i])) {
                int j = st.back();
                st.pop_back();
                res = max(res, n[j] * (dp[i] - dp[st.empty() ? 0 : st.back() + 1]));
            }
            st.push_back(i);
        }
        return res % 1000000007;
    }
};
```

## [Concatenated Words](https://leetcode.com/problems/concatenated-words)

A: 用set存储所有word便于查找，对每个单独的word，其是否是目标，可转换为dp问题。

```cpp
class Solution {
public:
    vector<string> findAllConcatenatedWordsInADict(vector<string>& words) {
        unordered_set<string> words_set;
        for (string word : words) words_set.insert(word);
        vector<string> res;
    
        for (string word : words) {
            int n = word.size();
            vector<int> dp(n + 1, 0);
            dp[0] = 1;
            for (int i = 0; i < n; i++) {
                if (!dp[i]) continue;
                for (int j = i + 1; j <= n; j++) {
                    if (j - i < n && words_set.count(word.substr(i, j - i))) {
                        dp[j] = 1;
                    }
                }
                if (dp[n]) {
                    res.push_back(word);
                    break;
                }
            }
        }
        return res;
    }
};
```

## [Minimum Cost For Tickets](https://leetcode.com/problems/minimum-cost-for-tickets)

A: 对全年天数进行dp。

```cpp
int mincostTickets(vector<int>& days, vector<int>& costs) {
  unordered_set<int> travel(begin(days), end(days));
  int dp[30] = {}; // 每三十天是一个费用循环
  for (int i = days.front(); i <= days.back(); ++i) {
    if (travel.find(i) == travel.end()) dp[i % 30] = dp[(i - 1) % 30];
    else dp[i % 30] = min({ dp[(i - 1) % 30] + costs[0],
        dp[max(0, i - 7) % 30] + costs[1], dp[max(0, i - 30) % 30] + costs[2] });
  }
  return dp[days.back() % 30];
}
```

A: 对days数组进行dp。

```cpp
int mincostTickets(vector<int>& days, vector<int>& costs, int cost = 0) {
  queue<pair<int, int>> last7, last30;
  for (auto d : days) {
    while (!last7.empty() && last7.front().first + 7 <= d) last7.pop();
    while (!last30.empty() && last30.front().first + 30 <= d) last30.pop();
    last7.push({ d, cost + costs[1] }); // 在该日购买7天票
    last30.push({ d, cost + costs[2] }); // 在该日购买月票
    cost = min({ cost + costs[0], last7.front().second, last30.front().second });
  }
  return cost;
}
```

## [Integer Break](https://leetcode.com/problems/integer-break)

A: DP，数组存储第i个数字的最大product。

```cpp
class Solution {
public:
    int integerBreak(int n) {
        
        if (n <= 2)
            return 1;

        vector<int> maxArr(n+1, 0);
                    
        /** For a number i: write i as a sum of integers, then take the product of those integers.
        maxArr[i] = maximum of all the possible products */
        
        maxArr[1] = 0;
        maxArr[2] = 1; // 2=1+1 so maxArr[2] = 1*1
        
        for (int i=3; i<=n; i++) {
            for (int j=1; j<i; j++) {
                /** Try to write i as: i = j + S where S=i-j corresponds to either one number or a sum of two or more numbers
                
                Assuming that j+S corresponds to the optimal solution for maxArr[i], we have two cases:
                (1) i is the sum of two numbers, i.e. S=i-j is one number, and so maxArr[i]=j*(i-j)
                (2) i is the sum of at least three numbers, i.e. S=i-j is a sum of at least 2 numbers,
                and so the product of the numbers in this sum for S is maxArr[i-j]
                (=maximum product after breaking up i-j into a sum of at least two integers):
                maxArr[i] = j*maxArr[i-j]
                */
                maxArr[i] = max(maxArr[i], max(j*(i-j), j*maxArr[i-j]));
            }
        }
        return maxArr[n];
    }
};
```

## [Number of Longest Increasing Subsequence](https://leetcode.com/problems/number-of-longest-increasing-subsequence)

A: DP。

[详解](https://leetcode.com/problems/number-of-longest-increasing-subsequence/solutions/107293/java-c-simple-dp-solution-with-explanation/?orderBy=most_votes)

```cpp
class Solution {
public:
    int findNumberOfLIS(vector<int>& nums) {
        int n = nums.size(), res = 0, max_len = 0;
        vector<pair<int,int>> dp(n,{1,1});            //dp[i]: {length, number of LIS which ends with nums[i]}
        for(int i = 0; i<n; i++){
            for(int j = 0; j <i ; j++){
                if(nums[i] > nums[j]){
                    if(dp[i].first == dp[j].first + 1)dp[i].second += dp[j].second; // ... j, i
                    if(dp[i].first < dp[j].first + 1)dp[i] = {dp[j].first + 1, dp[j].second};
                }
            }
            if(max_len == dp[i].first)res += dp[i].second;
            if(max_len < dp[i].first){
                max_len = dp[i].first;
                res = dp[i].second;
            }
        }
        return res;
    }
};
```

## [Best Team With No Conflicts](https://leetcode.com/problems/best-team-with-no-conflicts)

A: 按照年龄降序排列，此时对每个player，只需加入比他年纪大且得分高的分数即可。

```cpp
class Solution {
public:
    int bestTeamScore(vector<int>& scores, vector<int>& ages) {
        vector<pair<int, int>> players;
        int n = scores.size();
        for (int i=0; i<n; i++) {
            players.push_back({ages[i], scores[i]});
        }
        sort(players.begin(), players.end(), greater<>());
        
        int ans = 0;
        vector<int> dp(n);
        for (int i=0; i<n; i++) {
            int score = players[i].second;
            dp[i] = score;
            for (int j=0; j<i; j++) {
                if (players[j].second >= players[i].second) { // age of j is certainly >= i, so only important part to check 
                                                                //  before we add i and j in the same team is the score.
                    dp[i] = max(dp[i], dp[j] + score);
                }
            }
            ans = max(ans, dp[i]);
        }
        return ans;
    }
};
```

## [Stickers to Spell Word](https://leetcode.com/problems/stickers-to-spell-word)

A: DP，积累字符直到满足target。

```cpp
class Solution {
public:
    int minStickers(vector<string>& stickers, string target) {
        int m = stickers.size();
        vector<vector<int>> mp(m, vector<int>(26, 0));
        unordered_map<string, int> dp;
        // count characters a-z for each sticker 
        for (int i = 0; i < m; i++) 
            for (char c:stickers[i]) mp[i][c-'a']++;
        dp[""] = 0;
        return helper(dp, mp, target);
    }
private:
    int helper(unordered_map<string, int>& dp, vector<vector<int>>& mp, string target) {
        if (dp.count(target)) return dp[target];
        int ans = INT_MAX, n = mp.size();
        vector<int> tar(26, 0);
        for (char c:target) tar[c-'a']++;
        // try every sticker
        for (int i = 0; i < n; i++) {
            // optimization
            if (mp[i][target[0]-'a'] == 0) continue; 
            string s;
            // apply a sticker on every character a-z
            for (int j = 0; j < 26; j++) 
                if (tar[j]-mp[i][j] > 0) s += string(tar[j]-mp[i][j], 'a'+j);
            int tmp = helper(dp, mp, s); // s: 剩余字符
            if (tmp != -1) ans = min(ans, 1+tmp);
        }
        dp[target] = ans == INT_MAX? -1:ans;
        return dp[target];
    }
};
```

## [Number of Ways to Earn Points](https://leetcode.com/problems/number-of-ways-to-earn-points)

A: DP，背包问题。

```go
func waysToReachTarget(target int, types [][]int) int {
    ways := make([]int, target + 1)
    mod := 1000000007

    ways[0] = 1

    for _, t := range types {
        count, marks := t[0], t[1]

        for val := target; val > 0; val-- {
            total := 0

            for i := 0; i < count; i++ {
                total += marks
                if total > val { break }
                ways[val] += ways[val - total]
                ways[val] %= mod
            }
        }
    }

    return ways[target]
}
```

## [Fibonacci Number](https://leetcode.com/problems/fibonacci-number)

A: DP，记录前两个数。

```go
func fib(n int) int {
    if n == 0 {
        return 0
    }
    if n == 1 {
        return 1
    }
    first, second, cur := 0, 1, 1
    for n > 2 {
        first = second
        second = cur
        cur = first + second
        n--
    }
    return cur
}
```
