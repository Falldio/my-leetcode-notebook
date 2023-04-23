# Arrays & Hashing

## [Constains Duplicate](https://leetcode.com/problems/contains-duplicate/)

A: 使用哈希表存储数组元素，判断元素是否已经在表内。

```cpp
class Solution {
public:
    bool containsDuplicate(vector<int>& nums) {
        unordered_set<int> s;
        
        for (int i = 0; i < nums.size(); i++) {
            if (s.find(nums[i]) != s.end()) {
                return true;
            }
            s.insert(nums[i]);
        }
        
        return false;
    }
};
```

A: 空结构体可节约内存空间，相比于int或者bool。

```go
func containsDuplicate(nums []int) bool {
    set := make(map[int]struct{})
    for _, num := range nums {
        if _, hasNum := set[num]; hasNum {
            return true
        }
        set[num] = struct{}{}
    }
    return false
}
```

## [Valid Anagram](https://leetcode.com/problems/valid-anagram)

A: 建立哈希表：全为小写英文字母则用数组，反之用`unordered_map`；而后用哈希表数值加一减一判断条件是否成立。

```c++
class Solution {
public:
    bool isAnagram(string s, string t) {
        if (s.length() != t.length()) return false;
        int n = s.length();
        int counts[26] = {0};
        for (int i = 0; i < n; i++) { 
            counts[s[i] - 'a']++;
            counts[t[i] - 'a']--;
        }
        for (int i = 0; i < 26; i++)
            if (counts[i]) return false;
        return true;
    }
};
```

```go
func isAnagram(s string, t string) bool {
    if len(s) != len(t) {
        return false
    }
    counts := make([]int, 26)
    for i := 0; i < len(s); i++ {
        counts[s[i]-'a']++
        counts[t[i]-'a']--
    }
    for _, count := range counts {
        if count != 0 {
            return false
        }
    }
    return true
}
```

A: 快排，变乱序为有序。

```c++
class Solution {
public:
    bool isAnagram(string s, string t) { 
        sort(s.begin(), s.end());
        sort(t.begin(), t.end());
        return s == t; 
    }
};
```

## [Two Sum](https://leetcode.com/problems/two-sum)

A: 哈希表存储数值与索引，每次从哈希表中寻找是否有差值元素存在。

```c++
vector<int> twoSum(vector<int>& nums, int target) {
    unordered_map<int, int> imap;
    
    for (int i = 0;; ++i) {
        auto it = imap.find(target - nums[i]);
        
        if (it != imap.end()) 
            return vector<int> {i, it->second};
            
        imap[nums[i]] = i;
    }
}
```

```go
func twoSum(nums []int, target int) []int {
    s := make(map[int]int)
    for k, v := range nums {
        if _, ok := s[target-v]; ok {
            return []int{k, s[target-v]}
        }
        s[v] = k
    }
    return []int{}
}
```

## [Group Anagrams](https://leetcode.com/problems/group-anagrams/)

A: 按排序结果分组，将哈希表转换为双重数组。

```c++
vector<vector<string>> groupAnagrams(vector<string>& strs) {
    int n = strs.size();
    unordered_map<string, vector<string>> map;
    vector<vector<string>> ret;
    for (const auto& s : strs) {
        string t = s;
        sort(t.begin(), t.end());
        map[t].push_back(s);
    }
    ret.reserve(map.size());
    for (auto& p : map) {
        ret.push_back(std::move(p.second));
    }
    return ret;
}
```

## [Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements)

A: 哈希表统计频次，堆排序。

堆heap是一种数据结构，是一棵完全二叉树且满足性质：所有非叶子结点的值均不大于或均不小于其左、右孩子结点的值。

```cpp
class Solution {
public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        int n = nums.size();
        
        unordered_map<int, int> m;
        for (int i = 0; i < n; i++) {
            m[nums[i]]++;
        }
        
        vector<vector<int>> buckets(n + 1);
        for (auto it = m.begin(); it != m.end(); it++) {
            buckets[it->second].push_back(it->first);
        }
        
        vector<int> result;
        
        for (int i = n; i >= 0; i--) {
            if (result.size() >= k) {
                break;
            }
            if (!buckets[i].empty()) {
                result.insert(result.end(), buckets[i].begin(), buckets[i].end());
            }
        }
        
        return result;
    }
};
```

```go
func topKFrequent(nums []int, k int) []int {
    map_num:=map[int]int{}
    //记录每个元素出现的次数
    for _, item := range nums{
        map_num[item]++
    }
    h := &IHeap{}
    heap.Init(h) // Golang中的堆是一个接口，需要实现 Len, Less, Swap, Push, Pop 方法
    //所有元素入堆，堆的长度为k
    for key, value := range map_num {
        heap.Push(h, [2]int{key, value})
        if h.Len() > k {
            heap.Pop(h)
        }
    }
    res := make([]int,k)
    //按顺序返回堆中的元素
    for i := 0; i < k; i++ {
        res[k-i-1] = heap.Pop(h).([2]int)[0]
    }
    return res
}

//构建小顶堆
type IHeap [][2]int

func (h IHeap) Len()int {
    return len(h)
}

func (h IHeap) Less(i, j int) bool {
    return h[i][1] < h[j][1]
}

func (h IHeap) Swap(i, j int) {
    h[i], h[j] = h[j], h[i]
}

func (h *IHeap) Push(x interface{}) {
    *h = append(*h, x.([2]int))
}
func (h *IHeap) Pop() interface{} {
    old := *h
    n := len(old)
    x := old[n-1]
    *h = old[0:n-1]
    return x
}
```

## [Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self)

A: 同时计算前后缀乘积，遍历过程中更新前缀、后缀和答案。

```cpp
class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        vector<int> ans(size(nums),1);
        for(int i = 0, suf = 1, pre = 1, n = size(nums); i < n; i++) {
            ans[i] *= pre;             
            pre *= nums[i];
            ans[n-1-i] *= suf;
            suf *= nums[n-1-i];
        }
        return ans;
    }
};
```

## [Encode And Decode Strings](https://www.lintcode.com/problem/659/)

A: 长度+分隔符+原字符串。

```cpp
class Solution {
public:
    /*
     * @param strs: a list of strings
     * @return: encodes a list of strings to a single string.
     */
    string encode(vector<string> &strs) {
        // write your code here
        string res;
        for(auto str : strs){
            res += to_string(str.size()) + "#" + str;
        }
        return res;
    }

    /*
     * @param str: A string
     * @return: decodes a single string to a list of strings
     */
    vector<string> decode(string &str) {
        // write your code here
        vector<string> res;
        int i = 0;
        while(i < str.size()){
            int j = i;
            while(str[j] != '#')
            {
                j += 1;
            }
            int length = str_to_int(str.substr(i, j - i));
            res.push_back(str.substr(j + 1, length));
            i = j + length + 1;
        }
        return res;
    }

    int str_to_int(string str)
    {
        int res = 0;
        for(int i = 0; i < str.size(); i++){
            int num = str[i] - '0';
            res = res * 10 + num;
        }
        return res;
    }
};
```

## [Longest Consective Sequence](https://leetcode.com/problems/longest-consecutive-sequence)

A: 排序、去重、下标之差为数值之差。

```cpp
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        if (nums.size() == 0) return 0;
        sort(nums.begin(), nums.end());
        const vector<int>::const_iterator uidx = unique(nums.begin(), nums.end());
        nums.erase(uidx, nums.end());
        int len = nums.size();
        int ans = 1;
        int start = 0;
        for (int i = 1; i < len; i++) {
            if (nums[i] - nums[start] != i - start) {
                ans = max(i - start, ans);
                start = i;
            }
            if (i == len - 1) {
                ans = max(i - start + 1, ans);
            }
        }
        return ans;
    }
};
```

A: vector转unordered_set，从大小两边消除set中的连续数字。

```cpp
class Solution {
public:
    int longestConsecutive(vector<int> &num) {
        unordered_set<int> record(num.begin(),num.end());
        int res = 1;
        for (int n : num){
            if (record.find(n)==record.end()) continue;
            record.erase(n);
            int prev = n-1,next = n+1;
            while(record.find(prev)!=record.end()) record.erase(prev--);
            while(record.find(next)!=record.end()) record.erase(next++);
            res = max(res,next-prev-1);
        }
        return res;
    }
};
```

## [Maximum Product Subarray](https://leetcode.com/problems/maximum-product-subarray)

A: 维护当前两个最值，最小值应对负值乘积。

```cpp
class Solution {
public:
    int maxProduct(vector<int>& nums) {
        int currMax = nums[0];
        int currMin = nums[0];
        int result = nums[0];
        
        for (int i = 1; i < nums.size(); i++) {
            int temp = currMax;
            
            currMax = max(max(currMax * nums[i], currMin * nums[i]), nums[i]);
            currMin = min(min(currMin * nums[i], temp * nums[i]), nums[i]);
            
            result = max(result, currMax);
        }
        
        return result;
    }
};
```

## [Valid Sudoku](https://leetcode.com/problems/valid-sudoku)

A: 用三个矩阵分别存储行、列和子区域的数字状态。

```cpp
class Solution {
public:
    bool isValidSudoku(vector<vector<char>>& board) {
        const int cnt = 9;
        bool row[cnt][cnt] = {false};
        bool col[cnt][cnt] = {false};
        bool sub[cnt][cnt] = {false};
        
        for(int r = 0; r < cnt; ++r){
            for(int c = 0; c < cnt; ++c){
                if(board[r][c] == '.')
                    continue; // if not number pass
                
                int idx = board[r][c] - '0' - 1; //char to num idx
                int area = (r/3) * 3 + (c/3);
                
                // if number already exists
                if(row[r][idx] || col[c][idx] || sub[area][idx]){
                    return false;
                }
                
                row[r][idx] = true;
                col[c][idx] = true;
                sub[area][idx] = true;
            }
        }
        return true;
    }
};
```

## [Word Pattern](https://leetcode.com/problems/word-pattern)

A: 维持两组映射，检查映射关系和字符数目。

```cpp
bool wordPattern(string pattern, string str) {
    map<char, int> p2i;
    map<string, int> w2i;
    istringstream in(str);
    int n = pattern.size(), i = n;
    for (string word; in >> word; --i) {
        if (!i || p2i[pattern[n-i]] != w2i[word])
            return false;
        p2i[pattern[n-i]] = w2i[word] = i;
    }
    return !i; // 此时i正常情况为1
}
```

## [Replace Elements with Greatest Element on Right Side](https://leetcode.com/problems/replace-elements-with-greatest-element-on-right-side)

A: 反向遍历，保留当前最大元素。

```cpp
class Solution {
public:
    vector<int> replaceElements(vector<int>& arr) {
       int len = arr.size();
       vector<int> ans(len);
       ans[len - 1] = -1;
       for (int i = len - 2; i >= 0; i--) {
           ans[i] = max(ans[i + 1], arr[i + 1]);
       }
       return ans;
    }
};
```

## [Is Subsequence](https://leetcode.com/problems/is-subsequence)

A: 两个指针分别在两个字符串上滑动，最后检查指针位置。

```cpp
class Solution {
public:
    bool isSubsequence(string s, string t) {
        int len1 = s.size(), len2 = t.size();
        int i = 0;
        for (int j = 0; j < len2; j++) {
            if (t[j] == s[i]) i++;
        }
        return i == len1;
    }
};
```

A: DP。

```go
func isSubsequence(s string, t string) bool {
    if len(s) > len(t) {
        return false
    }
    pre := make([]int, len(t) + 1)
    cur := make([]int, len(t) + 1)
    for i := 1; i <= len(s); i++ {
        for j := 1; j <= len(t); j++ {
            if s[i - 1] == t[j - 1] {
                cur[j] = 1 + pre[j - 1]
            } else {
                cur[j] = max(pre[j], cur[j - 1])
            }
        }
        tmp := make([]int, len(t) + 1)
        copy(tmp, cur)
        pre = tmp
    }
    return pre[len(t)] == len(s)
}

func max(i, j int) int {
    if i > j {
        return i
    } else {
        return j
    }
}
```

## [Lexicographically Smallest Equivalent String](https://leetcode.com/problems/lexicographically-smallest-equivalent-string)

A: 并查集。

```cpp
class Solution {
public:
    int par[26];
    
    int find(int x){
        if(par[x]==-1) return x;
        return par[x]=find(par[x]);
    }
    
    void Union(int x, int y) {
        x = find(x);
        y = find(y);
        
        if (x != y) 
            par[max(x, y)] = min(x, y); 
    }

    string smallestEquivalentString(string s1, string s2, string baseStr) {
        
        memset(par, -1, sizeof(par));
        
        for (auto i = 0; i < s1.size(); ++i) 
            Union(s1[i] - 'a', s2[i] - 'a');
        
        for(auto i=0;i<baseStr.size();i++) 
            baseStr[i]=find(baseStr[i]-'a')+'a';

        return baseStr;
    }
};
```

## [Sort Colors](https://leetcode.com/problems/sort-colors)

A: 存储颜色和该颜色对应数量，依次改写nums。

```cpp
class Solution {
public:
    void sortColors(vector<int>& nums) {
        unordered_map<int, int> c;
        for (auto n : nums) c[n]++;
        int cur = 0;
        for (auto &[color, num] : c) {
            for (int i = 0; i < num; i++) {
                nums[cur + i] = color;
            }
            cur += num;
        }
    }
};
```

## [Maximum Sum Circular Subarray](https://leetcode.com/problems/maximum-sum-circular-subarray)

A: [Maximum Subarray](https://leetcode.com/problems/maximum-subarray)的延伸，考虑目标子数组分布在原数组两端的特殊情况。

```cpp
int maxSubarraySumCircular(vector<int>& A) {
    int total = 0, maxSum = A[0], curMax = 0, minSum = A[0], curMin = 0;
    for (int& a : A) {
        curMax = max(curMax + a, a);
        maxSum = max(maxSum, curMax);
        curMin = min(curMin + a, a);
        minSum = min(minSum, curMin);
        total += a;
    }
    // maxSum <= 0 则数组无正数，返回最大非正数。
    // maxSum > 0 则考虑两种情况：1）目标子数组在原数组中间，直接返回maxSum 2）目标子数组在原数组两端，则中间部分为minSum
    return maxSum > 0 ? max(maxSum, total - minSum) : maxSum;
}
```

## [Encode And Decode Tinyurl](https://leetcode.com/problems/encode-and-decode-tinyurl)

A: 存储双向的键值对以便查询url。

```cpp
class Solution {
public:
    unordered_map<string, string> codeDB, urlDB;
    const string chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    
    string getCode() {
        string code = "";
        for (int i = 0; i < 6; i++) code += chars[rand() % 62];
        return "http://tinyurl.com/" + code;
    }
    
    string encode(string longUrl) {
        if (urlDB.find(longUrl) != urlDB.end()) return urlDB[longUrl];
        string code = getCode();
        while (codeDB.find(code) != codeDB.end()) code = getCode();
        codeDB[code] = longUrl;
        urlDB[longUrl] = code;
        return code;
    }

    string decode(string shortUrl) {
        return codeDB[shortUrl];
    }
};
```

## [Subarray Sums Divisible by K](https://leetcode.com/problems/subarray-sums-divisible-by-k)

A: 前缀和。

[详解](https://leetcode.com/problems/subarray-sums-divisible-by-k/solutions/413234/detailed-whiteboard-beats-100-do-you-really-want-to-understand-it/?orderBy=most_votes)

```cpp
class Solution {
public:
    int subarraysDivByK(vector<int>& A, int K) {
        vector<int> count(K); // 前缀和数组，存储0 - K-1的个数
        count[0] = 1;
        int prefix = 0, res = 0;
        for (int a : A) {
            prefix = (prefix + a % K + K) % K; // 当前和与K的余数
            res += count[prefix]++; // 如果该前缀已经找到过（之前i位置，现在j位置），则说明i到j为一个有效数组，因为前缀和不变，中间的抵消了
        }
        return res;
    }
};
```

## [Brick Wall](https://leetcode.com/problems/brick-wall)

A: 统计边缘个数。

```cpp
class Solution {
public:
    int leastBricks(vector<vector<int>>& wall) 
    {
        unordered_map<int, int> edge_frequency;     //HashMap to store the number of common edges among the rows
        int max_frequency = 0;         //Variable to store the frequency of most occuring edge
        
        for(int row=0; row<wall.size(); row++)        //Iterating through each row
        {
            int edge_postion = 0;       //Variable to store different edge postion
            
            for(int brick_no=0; brick_no< wall[row].size() -1; brick_no++)    //Iterating through each brick inside a row
            { 
                int current_brick_length = wall[row][brick_no];  //Length of the current brick
                edge_postion = edge_postion + current_brick_length ;  //Next Edge Position = Previous Edge Position + Current Brick's Length
                edge_frequency[edge_postion]++;  //Incrementing the Frequency of just calculated Edge Postion
                max_frequency = max(edge_frequency[edge_postion],max_frequency);  //Comparing the "Frequency of just calculated Edge Postion" with "Max Frequency seen till now" & storing whichever is greater.
            }
        }
        return wall.size() - max_frequency; // returning (Number of Bricks Crossed by Line) i.e. (Number of Rows in Wall - Frequency of Most Occuring Edge) 
    }
};
```

## [Best Time to Buy And Sell Stock II](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/)

A: 一旦有利可图则卖出股票。

```cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int ans =0;
        
        // look into the next day and if you are making profit just buy it today.
        for(int i =1;i<prices.size();i++){
            if(prices[i]>prices[i-1]) ans += (prices[i]-prices[i-1]);
        }
        return ans;
    }
};
```

A: 单调队列。

```go
func maxProfit(prices []int) int {
    q := []int{}
    ans := 0
    for _, p := range prices {
        if len(q) == 0 {
            q = append(q, p)
        } else {
            cur := q[len(q) - 1]
            if p < cur {
                // keep increasing order
                ans += cur - q[0]
                q = []int{}
            }
            q = append(q, p)
        }
    }
    if len(q) > 0 {
        ans += q[len(q) - 1] - q[0]
    }
    return ans
}
```

A: 贪心算法，计算前一天买今天卖的利润，收集所有的正利润。

```go
func maxProfit(prices []int) int {
    ans := 0
    profits := []int{}
    for i := 1; i < len(prices); i++ {
        profits = append(profits, prices[i] - prices[i - 1])
    }
    for _, v := range profits {
        if v > 0 {
            ans += v
        }
    }
    return ans
}
```

## [Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k)

A: 前缀和，用map记录前缀和，这样一旦有满足情况的，可以直接加入ans。

```cpp
class Solution {
public:
    int subarraySum(vector<int>& arr, int k) {
        int n = arr.size(); // take the size of the array
        
        int prefix[n]; // make a prefix array to store prefix sum
        
        prefix[0] = arr[0]; // for element at index at zero, it is same
        
        // making our prefix array
        for(int i = 1; i < n; i++)
        {
            prefix[i] = arr[i] + prefix[i - 1];
        }
        
        unordered_map<int,int> mp; // declare an unordered map
        
        int ans = 0; // to store the number of our subarrays having sum as 'k'
        
        for(int i = 0; i < n; i++) // traverse from the prefix array
        {
            if(prefix[i] == k) // if it already becomes equal to k, then increment ans
                ans++;
            
            // now, as we discussed find whether (prefix[i] - k) present in map or not
            if(mp.find(prefix[i] - k) != mp.end())
            {
                // 由于前缀和递增，不用考虑重复相加
                ans += mp[prefix[i] - k]; // if yes, then add it our answer
            }
            
            mp[prefix[i]]++; // put prefix sum into our map
        }
        
        return ans; // and at last, return our answer
    }
};
```

## [Unique Length-3 Palindromic Subsequences](https://leetcode.com/problems/unique-length-3-palindromic-subsequences)

A: 记录首次和末次出现某字母的位置，计算两者之间的字母种数即可得到这个字母为首尾组成的三位回文个数。

```cpp
int countPalindromicSubsequence(string s) {
    int first[26] = {[0 ... 25] = INT_MAX}, last[26] = {}, res = 0;
    for (int i = 0; i < s.size(); ++i) {
        first[s[i] - 'a'] = min(first[s[i] - 'a'], i);
        last[s[i] - 'a'] = i;
    }
    for (int i = 0; i < 26; ++i)
        if (first[i] < last[i])
            res += unordered_set<char>(begin(s) + first[i] + 1, begin(s) + last[i]).size();
    return res;
}
```

## [LFU Cache](https://leetcode.com/problems/lfu-cache)

A: 用三个哈希表分别存储{key, value, freq}, {key, 同频率key list}, {freq, key list}。当溢出时，通过min freq找到对应key list，删除其中的一个元素。

```cpp
class LFUCache {
    int cap;
    int size;
    int minFreq;
    unordered_map<int, pair<int, int>> m; //key to {value,freq};
    unordered_map<int, list<int>::iterator> mIter; //key to list iterator;
    unordered_map<int, list<int>>  fm;  //freq to key list; 相同频率下可能有多个key
public:
    LFUCache(int capacity) {
        cap=capacity;
        size=0;
    }
    
    int get(int key) {
        if(m.count(key)==0) return -1;
        
        fm[m[key].second].erase(mIter[key]);
        m[key].second++;
        fm[m[key].second].push_back(key);
        mIter[key]=--fm[m[key].second].end();
        
        if(fm[minFreq].size()==0 ) 
              minFreq++;
        
        return m[key].first;
    }
    
   void put(int key, int value) {
        if(cap<=0) return;
        
        int storedValue=get(key);
        if(storedValue!=-1)
        {
            m[key].first=value;
            return;
        }
        
        if(size>=cap)
        {
            m.erase( fm[minFreq].front() );
            mIter.erase( fm[minFreq].front() );
            fm[minFreq].pop_front();
            size--;
        }
        
        m[key]={value, 1};
        fm[1].push_back(key);
        mIter[key]=--fm[1].end();
        minFreq=1;
        size++;
    }
};
```

## [Minimum Number of Swaps to Make the String Balanced](https://leetcode.com/problems/minimum-number-of-swaps-to-make-the-string-balanced)

A: 双指针，统计未能成组的]数量。

[详解](https://leetcode.com/problems/minimum-number-of-swaps-to-make-the-string-balanced/solutions/1390576/two-pointers/)

```cpp
int minSwaps(string s) {
    int res = 0, bal = 0;
    for (int i = 0, j = s.size() - 1; i < j; ++i) {
        bal += s[i] == '[' ? 1 : -1;
        if (bal < 0) {
             // 出现孤儿括号]，则从右往左找到[
            for (int bal1 = 0; bal1 >= 0; --j)
                bal1 += s[j] == ']' ? 1 : -1;
            swap(s[i], s[j + 1]);
            ++res;
            bal = 1;
        }
    }
    return res;
}
```

## [Number of Pairs of Interchangeable Rectangles](https://leetcode.com/problems/number-of-pairs-of-interchangeable-rectangles)

A: 按照长宽比例统计结果，计算。

```cpp
class Solution {
public:
    long long interchangeableRectangles(vector<vector<int>>& rectangles) {
        long long result = 0;
        map<pair<int, int>, int> mp;

        for (auto& rect : rectangles) {
            int gcd = __gcd(rect[0], rect[1]);
            pair<int, int> key = {rect[0]/gcd, rect[1]/gcd};
            if(mp.find(key) != mp.end()) result += mp[key]; // 新加的矩形和相同长宽比的原有矩形各组成一对
            mp[key]++;
        }

        return result;
    }
};
```

## [Maximum Product of the Length of Two Palindromic Subsequences](https://leetcode.com/problems/maximum-product-of-the-length-of-two-palindromic-subsequences)

A: 分三种情况回溯，s[i]分别在两个回文中、不选择s[i]。

```cpp
class Solution {
public:
    int result = 0;
    bool isPalin(string& s){
        int i = 0;
        int j = s.length() - 1;
 
        while (i < j) {
            if (s[i] != s[j]) return false;
            i++;
            j--;
        }
 
        return true;
    }
    
    void dfs(string& s, int i, string& s1, string& s2){
        
        if(i >= s.length()){
            if(isPalin(s1) && isPalin(s2))
                result = max(result, (int)s1.length()*(int)s2.length());
            return;
        }
        
        s1.push_back(s[i]);
        dfs(s, i+1, s1, s2);
        s1.pop_back();
        
        s2.push_back(s[i]);
        dfs(s, i+1, s1, s2);
        s2.pop_back();
        
        dfs(s, i+1, s1, s2);
    }
    
    int maxProduct(string s) {
        string s1 = "", s2 = "";
        dfs(s, 0, s1, s2);
        
        return result;
    }
};
```

## [Greatest Common Divisor of Strings](https://leetcode.com/problems/greatest-common-divisor-of-strings)

A: 若两个字符串相加为回文，则divider长度为两字符串长度最大公因数，反之为空。

```cpp
class Solution {
public:
    string gcdOfStrings(const string& s1, const string& s2)
    {
        return (s1 + s2 == s2 + s1)  
        ? s1.substr(0, gcd(size(s1), size(s2)))
            : "";
    }
};
```

## [Grid Game](https://leetcode.com/problems/grid-game)

A: robot1有n种路径（在第一行i处向下），topSum和bottomSum为robot1未能归零的格子，即robot2得分的两种情况。

[详解](https://leetcode.com/problems/grid-game/solutions/1486340/c-java-python-robot1-minimize-topsum-and-bottomsum-of-robot-2-picture-explained/?orderBy=most_votes)

DP不能用于求解该问题

```cpp
class Solution {
public:
    long long gridGame(vector<vector<int>>& grid) {
        long long topSum = accumulate(begin(grid[0]), end(grid[0]), 0LL), bottomSum = 0;
        long long ans = LLONG_MAX;
        for (int i = 0; i < grid[0].size(); ++i) {
            topSum -= grid[0][i];
            ans = min(ans, max(topSum, bottomSum));
            bottomSum += grid[1][i];
        }
        return ans;
    }
};
```

## [Find All Anagrams in a String](https://leetcode.com/problems/find-all-anagrams-in-a-string)

A: 滑动窗口，比较p和窗口内字母频率。

```cpp
class Solution {
public:
    vector<int> findAnagrams(string s, string p) {
        int s_len = s.length();
        int p_len = p.length();
        
        if(s.size() < p.size()) return {};
        
        vector<int> freq_p(26,0);
        vector<int> window(26,0);
        
        //first window
        for(int i=0;i<p_len;i++){
            freq_p[p[i]-'a']++;
            window[s[i]-'a']++;
        }
        
        vector<int> ans;
        if(freq_p == window) ans.push_back(0);
        
        for(int i=p_len;i<s_len;i++){
            window[s[i-p_len] - 'a']--;
            window[s[i] - 'a']++;
            
            if(freq_p == window) ans.push_back(i-p_len+1);
        }
        return ans;
    }
};
```

```go
func findAnagrams(s string, p string) []int {
    m1 := map[byte]int{}
    m2 := map[byte]int{}
    ans := []int{}
    for i := range p {
        m1[p[i]]++
    }
    l, r, valid := 0, 0, 0
    for r < len(s) {
        if _, ok := m1[s[r]]; ok {
            m2[s[r]]++
            if m1[s[r]] == m2[s[r]] {
                valid++
            }
        }
        r++
        if r - l == len(p) {
            if valid == len(m1) {
                ans = append(ans, l)
            }
            if _, ok := m1[s[l]]; ok {
                if m1[s[l]] == m2[s[l]] {
                    valid--
                }
                m2[s[l]]--
            }
            l++
        }
    }
    return ans
}
```

## [Find the Index of the First Occurrence in a String](https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string)

A: [KMP算法](http://jakeboxer.com/blog/2009/12/13/the-knuth-morris-pratt-algorithm-in-my-own-words/)。

```cpp
class Solution {
public:
    int strStr(string haystack, string needle) {
        int m = haystack.size(), n = needle.size();
        if (!n) {
            return 0;
        }
        vector<int> lps = kmpProcess(needle);
        for (int i = 0, j = 0; i < m;) {
            if (haystack[i] == needle[j]) { 
                i++, j++;
            }
            if (j == n) {
                return i - j;
            }
            if (i < m && haystack[i] != needle[j]) {
                j ? j = lps[j - 1] : i++;
            }
        }
        return -1;
    }
private:
    vector<int> kmpProcess(string needle) {
        int n = needle.size();
        vector<int> lps(n, 0);
        for (int i = 1, len = 0; i < n;) {
            if (needle[i] == needle[len]) {
                lps[i++] = ++len;
            } else if (len) {
                len = lps[len - 1];
            } else {
                lps[i++] = 0;
            }
        }
        return lps;
    }
};
```

```go
func strStr(haystack string, needle string) int {
    len1, len2 := len(haystack), len(needle)
    if len2 > len1 {
        // impossible cases
        return -1
    }
    var pmt = kmp(needle) // init partial match table
    for i, j := 0, 0; i < len1; {
        if haystack[i] == needle[j] {
            // if the two strings match each other partially
            i++; j++
        }
        if j == len2 { // first occurence
            return i - j
        }
        if i < len1 && haystack[i] != needle[j] {
            // the two strings differ
            if j != 0 {
                // try another prefix
                j = pmt[j - 1]
            } else {
                i++
            }
        }
    }
    return -1
}

func kmp(needle string) []int {
    size := len(needle)
    pmt := make([]int, size) // partial match table {idx, max partial match prefix length}
    for i, len := 1, 0; i < size; {
        // len means the longest proper prefix length
        // i starts from 1 because there's no prefix at pos 0
        if needle[i] == needle[len] {
            // the longest prefix([0, i - 1]) matches
            pmt[i] = len + 1
            i++; len++
        } else if len != 0 {
            // if they don't match, check if another sub-prefix([0, i - 2]) matches
            len = pmt[len - 1]
        } else {
            // if there's no more prefix availiable
            pmt[i] = 0
            i++
        }
    }
    return pmt
}
```

## [Wiggle Sort](https://www.lintcode.com/problem/508)

A: 排序后每两个数字交换顺序。

```cpp
class Solution {
public:
    /*
     * @param nums: A list of integers
     * @return: nothing
     */
    void wiggleSort(vector<int> &nums) 
    {
        sort(nums.begin(),nums.end());
        for(int i=2;i<nums.size();i+=2)
        {
            int temp = nums[i];
            nums[i] = nums[i-1];
            nums[i-1] = temp;
        }
    }
};
```

## [Largest Number](https://leetcode.com/problems/largest-number)

A: 通过局部两个数字的组合来判断这两个数字的顺序。

```cpp
class Solution {
public:
    string largestNumber(vector<int> &num) {
        vector<string> arr;
        for(auto i:num)
            arr.push_back(to_string(i));
        sort(begin(arr), end(arr), [](string &s1, string &s2){ return s1+s2>s2+s1; });
        string res;
        for(auto s:arr)
            res+=s;
        while(res[0]=='0' && res.length()>1)
            res.erase(0,1);
        return res;
    }
};
```

## [Continuous Subarray Sum](https://leetcode.com/problems/continuous-subarray-sum)

A: 哈希表保存前缀和的余数，当出现两次相同余数时，则说明两者之差为结果数组。

```cpp
class Solution {
public:
    bool checkSubarraySum(vector<int>& nums, int k) {
        
        //Questions to ask the interviewer - 
        
        //1. So you said k is an integer? Can the k be equal to 0? Can it be n-ve?
        // ANS - k can be positive, zero or negative - consider all cases !
        
        // Positive case - [23, 2, 4, 6, 7],  k=6; ANS = true
        // Negative Case - [23,2,4,6,7], k= -6; ANS = true (Since n=-1 and -1*-6=6)
        // Zero Case - [0,0], k=0; ANS = true 
        
        //2. 'n' can be anything right? positive, negative and zero
        
        //Explanation of algorithm to interviewer - 
        
        //A proof sketch:
        // Suppose sum_i represents the running sum starting from index 0 and ending at i
        // once we find a mod that has been seen, say modk, we have:
        
        // current one: sum_i = m*k + modk
        // previous one: sum_j = n*k + modk
        // Thus,
        // sum_i - sum_j = (m - n) *k
        
        //so if two runningSum mod k have the same values, then when they are subtracted, they are bound to be multiples of k

        //base checking - first check if the size of the array is less than 2
        
        if(nums.size()<2)
            return false;
        
        //Create a hashmap of the running_sum remainder and it's respective index
        
        unordered_map<int, int> mp;
        
        //Why to insert <0,-1> for the hashmap
        
        // <0,-1> can allow it to return true when the runningSum%k=0,
        
        //for example [1,2,3] is input and k=6
        //then the remainders are [ 1,3,0] i.e runningSum = runningSum%k
        //now 1+2+3=6 which is actually a multiple of 6 and hence 0 should be stored in the hashmap
        
        //ok - but why -1?
        //-1 is good for storing for 0 because - it will remove the case where we consider only the first element which alone may be a multiple as 0-(-1) is not greater than 1
        
        // In addition, it also avoids the first element of the array is the multiple of k, since 0-(-1)=1 is not greater than 1.
        
        mp[0]=-1;
        
        int runningSum=0;
        
        for(int i=0;i<nums.size();i++)
        {
            runningSum+=nums[i];
            
            if(k!=0) 
                runningSum = runningSum%k;
            
            //check if the runningsum already exists in the hashmap
            if(mp.find(runningSum)!=mp.end())
            {
                //if it exists, then the current location minus the previous location must be greater than 1
                
                if(i-mp[runningSum]>1)
                    return true;
            }
            else
            {
                //otherwise if the current runningSum doesn't exist in the hashmap, then store it as it maybe used later on
                
                mp[runningSum]=i;
            }
                    
        }
        
        return false;
        
    }
};
```

## [Push Dominoes](https://leetcode.com/problems/push-dominoes)

A: 双指针。

```cpp
string pushDominoes(string d) {
    d = 'L' + d + 'R';
    string res = "";
    for (int i = 0, j = 1; j < d.length(); ++j) {
        if (d[j] == '.') continue;
        int middle = j - i - 1; // i， j之间字符数
        if (i > 0)
            res += d[i];
        if (d[i] == d[j])
            res += string(middle, d[i]);
        else if (d[i] == 'L' && d[j] == 'R') // i. j向相反方向倒
            res += string(middle, '.');
        else // i, j向同方向倒
            res += string(middle / 2, 'R') + string(middle % 2, '.') + string(middle / 2, 'L');
        i = j;
    }
    return res;
}
```

## [Repeated DNA Sequences](https://leetcode.com/problems/repeated-dna-sequences)

A: [详解](https://leetcode.com/problems/repeated-dna-sequences/solutions/53952/20-ms-solution-c-with-explanation/)。本质为用位存储检索状态以减小空间复杂度，自定义哈希函数以替代map。

```cpp
class Solution {
public:
    vector<string> findRepeatedDnaSequences(string s) {
        if (s.size() <= 10)
            return vector<string>();

        vector<string> R;
        // 最大可能子字符串数目为1<<20
        bitset<1<<20> S1; // string是否被检索过
        bitset<1<<20> S2; // string出现次数是否大于1

        int val = 0;
        for (int i=0; i < 10; i++)   // Calc. the hash value for the first string.
            val = (val << 2) | char2val(s[i]);
        S1.set(val);

        int mask = (1 << 20) - 1;
        for (int i=10; i < s.size(); i++) {
            // Calc the hash value for the string ending at position i.
            val = ((val << 2) & mask) | char2val(s[i]);  
            if (S2[val])
                continue;
            if (S1[val]) {
                R.push_back(s.substr(i-10+1, 10));
                S2.set(val);
            }
            else
                S1.set(val);
        }
        return R;
    }

    int char2val(char c) {
        switch (c) {
            case 'A': return 0;
            case 'C': return 1;
            case 'G': return 2;
            case 'T': return 3;
            default: return -1;
        }
    }
};
```

## [Insert Delete GetRandom O(1)](https://leetcode.com/problems/insert-delete-getrandom-o1)

A: map判断是否已出现该数字，vector用于随机获取元素。

```cpp
class RandomizedSet {
public:
    /** Initialize your data structure here. */
    RandomizedSet() {
        
    }
    
    /** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
    bool insert(int val) {
        if (m.find(val) != m.end()) return false;
        nums.emplace_back(val);
        m[val] = nums.size() - 1;
        return true;
    }
    
    /** Removes a value from the set. Returns true if the set contained the specified element. */
    bool remove(int val) {
        if (m.find(val) == m.end()) return false;
        int last = nums.back();
        m[last] = m[val];
        nums[m[val]] = last;
        nums.pop_back();
        m.erase(val);
        return true;
    }
    
    /** Get a random element from the set. */
    int getRandom() {
        return nums[rand() % nums.size()];
    }
private:
    vector<int> nums;
    unordered_map<int, int> m; // {val, index}
};
```

## [Check If a String Contains All Binary Codes of Size K](https://leetcode.com/problems/check-if-a-string-contains-all-binary-codes-of-size-k)

A: 收集所有的substr，其数量应该等于长度为k的二进制码总数。

```cpp
class Solution {
public:
    bool hasAllCodes(string s, int k) {
        if (k > s.size()) return false;
        
        unordered_set<string> my_set;
        
        for (int i = 0; i <= s.size()-k; i++)
            my_set.insert(s.substr(i, k));
        
        return my_set.size() == pow(2, k);
    }
};
```

## [Range Sum Query 2D - Immutable](https://leetcode.com/problems/range-sum-query-2d-immutable)

A: 二维前缀和。

```cpp
class NumMatrix {
private:
    int row, col;
    vector<vector<int>> sums;
public:
    NumMatrix(vector<vector<int>> &matrix) {
        row = matrix.size();
        col = row>0 ? matrix[0].size() : 0;
        sums = vector<vector<int>>(row+1, vector<int>(col+1, 0));
        for(int i=1; i<=row; i++) {
            for(int j=1; j<=col; j++) {
                sums[i][j] = matrix[i-1][j-1] + 
                             sums[i-1][j] + sums[i][j-1] - sums[i-1][j-1] ;
            }
        }
    }

    int sumRegion(int row1, int col1, int row2, int col2) {
        return sums[row2+1][col2+1] - sums[row2+1][col1] - sums[row1][col2+1] + sums[row1][col1];
    }
};
```

```go
type NumMatrix struct {
    prefix [][]int
}


func Constructor(matrix [][]int) NumMatrix {
    prefix := make([][]int, len(matrix) + 1)
    for i := range prefix {
        prefix[i] = make([]int, len(matrix[0]) + 1)
    }
    for i := 1; i < len(prefix); i++ {
        for j := 1; j < len(prefix[0]); j++ {
            prefix[i][j] += prefix[i - 1][j] + prefix[i][j - 1] - prefix[i - 1][j - 1] + matrix[i - 1][j - 1]
        }
    }
    return NumMatrix{prefix}
}


func (this *NumMatrix) SumRegion(row1 int, col1 int, row2 int, col2 int) int {
    return this.prefix[row2 + 1][col2 + 1] - this.prefix[row1][col2 + 1] - this.prefix[row2 + 1][col1] + this.prefix[row1][col1]
}


/**
 * Your NumMatrix object will be instantiated and called as such:
 * obj := Constructor(matrix);
 * param_1 := obj.SumRegion(row1,col1,row2,col2);
 */
```

## [Non-decreasing Array](https://leetcode.com/problems/non-decreasing-array)

A: 检测逆序数量，当发生逆序时调整元素值，使得逆序数量减少。如果逆序数量大于1，则返回false。

```cpp
class Solution {
public:
    bool checkPossibility(vector<int>& nums) {
        int cnt = 0;
        for (int i = 1; i < nums.size() && cnt <= 1; ++i) {
            if (nums[i - 1] > nums[i]) {
                ++cnt;
                if (i - 2 < 0 || nums[i - 2] <= nums[i]) nums[i - 1] = nums[i]; // modify nums[i-1] of a priority
                else nums[i] = nums[i - 1]; // have to modify nums[i]
            }
        }
        return cnt <= 1;
    }
};
```

## [First Missing Positive](https://leetcode.com/problems/first-missing-positive)

A: 首先将所有小于等于0的数都置为n+1(out of range)，然后遍历数组，将对应的数的下标位置的数置为负数，最后遍历数组，找到第一个大于0的数，其下标加1即为所求。

```cpp
class Solution {
public:
    int firstMissingPositive(vector<int>& nums) {
        int n = nums.size();
        for (int i = 0; i < n; i++) if (nums[i] <= 0) nums[i] = n + 1; // 将所有非正数置为n+1，即out of range
        for (int i = 0; i < n; i++) if (abs(nums[i]) <= n && nums[abs(nums[i]) - 1] > 0) nums[abs(nums[i]) - 1] *= -1; // abs(nums[i]) - 1为nums[i]的下标，将其置为负数 (nums[i]的正常值应该是i+1)
        for (int i = 0; i < n; i++) if (nums[i] > 0) return i + 1;
        return n + 1;
    }
};
```

## [Length of Last Word](https://leetcode.com/problems/length-of-last-word)

A: 从后往前遍历，先找到最后一个非空格字符，然后再找到第一个空格字符。

```cpp
class Solution {
public:
    int lengthOfLastWord(string s) { 
        int len = 0, tail = s.length() - 1;
        while (tail >= 0 && s[tail] == ' ') tail--;
        while (tail >= 0 && s[tail] != ' ') {
            len++;
            tail--;
        }
        return len;
    }
};
```

## [Longest Common Prefix](https://leetcode.com/problems/longest-common-prefix)

A: 排列字符串，然后比较第一个和最后一个字符串的公共前缀。

```cpp
class Solution {
public:
    string longestCommonPrefix(vector<string>& str) {
        int n = str.size();
        if(n==0) return "";
        
        string ans  = "";
        sort(begin(str), end(str));
        string a = str[0];
        string b = str[n-1];
        
        for(int i=0; i<a.size(); i++){
            if(a[i]==b[i]){
                ans = ans + a[i];
            }
            else{
                break;
            }
        }
        
        return ans;
        
    }
};
```

## [Pascal's Triangle](https://leetcode.com/problems/pascals-triangle)

A: DP，逐行计算。

```cpp
class Solution {
public:
    vector<vector<int> > generate(int numRows) {
        vector<vector<int>> r(numRows);

        for (int i = 0; i < numRows; i++) {
            r[i].resize(i + 1);
            r[i][0] = r[i][i] = 1;
  
            for (int j = 1; j < i; j++)
                r[i][j] = r[i - 1][j - 1] + r[i - 1][j];
        }
        
        return r;
    }
};
```

## [Remove Element](https://leetcode.com/problems/remove-element)

A: 双指针，j指向当前不等于val的位置，i遍历数组，如果nums\[i]不等于val，则将nums\[i]赋值给nums\[j]，然后j++。

```cpp
class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        int j=0;
        for(int i=0;i<nums.size();i++){
            if(nums[i]!=val){
                nums[j++]=nums[i];
            }
        }
        return j;        
    }
};
```

```go
func removeElement(nums []int, val int) int {
    i, j := 0, 0
    for i < len(nums) {
        if nums[i] != val {
            nums[j] = nums[i]
            j++
        }
        i++
    }
    return j
}
```

## [Unique Email Addresses](https://leetcode.com/problems/unique-email-addresses)

A: 用@分割字符串，然后将local部分中的.和+之后的部分去掉，最后将local和domain拼接起来，放入set中。

remove(): 将指定范围中等于第三个参数的元素用下一个未被删除的元素替换，返回指向最后一个未被删除元素的下一个位置的迭代器。

erase(): 删除指定范围中的元素，返回指向最后一个被删除元素的下一个位置的迭代器。

```cpp
class Solution {
public:
    int numUniqueEmails(vector<string>& emails) {
        unordered_set<string> s;
        for (auto email : emails) {
            int i = email.find('@');
            string local = email.substr(0, i), domain = email.substr(i);
            if (auto pos = local.find('+'); pos != string::npos) local = local.substr(0, pos);
            local.erase(remove(begin(local), end(local), '.'), end(local));
            s.insert(local + domain);
        }
        return s.size();
    }
};
```

## [Isomorphic Strings](https://leetcode.com/problems/isomorphic-strings)

A: rep保留s中每个字符的替换，used保留t中每个字符是否被记录过。

```cpp
class Solution {
public:
    bool isIsomorphic(string s, string t) {
        // Use hashmaps to save the replacement for every character in the first string...
        unordered_map <char , char> rep;
        unordered_map <char , bool> used;
        // Traverse all elements through the loop...
        for(int idx = 0 ; idx < s.length() ; idx++) {
            // If rep contains s[idx] as a key...
            if(rep.count(s[idx])) {
                // Check if the rep is same as the character in the other string...
                // If not, the strings can’t be isomorphic. So, return false...
                if(rep[s[idx]] != t[idx])
                    return false;
            }
            // If no replacement found for first character, check if the second character has been used as the replacement for any other character in the first string...
            else {
                if(used[t[idx]])
                    return false;
                // If there exists no character whose replacement is the second character...
                // Assign the second character as the replacement of the first character.
                rep[s[idx]] = t[idx];
                used[t[idx]] = true;
            }
        }
        // Otherwise, the strings are not isomorphic.
        return true;
    }
};
```

A: 用两个map分别保存s到t的映射和t到s的映射，如果无法映射，返回false。
```go
func isIsomorphic(s string, t string) bool {
	map1 := make(map[byte]byte)
	map2 := make(map[byte]byte)
	for i := range s {
		if _, ok := map1[s[i]]; !ok {
			map1[s[i]] = t[i] // map1保存 s[i] 到 t[j]的映射
		}
		if _, ok := map2[t[i]]; !ok {
			map2[t[i]] = s[i] // map2保存 t[i] 到 s[j]的映射
		}
		// 无法映射，返回 false
		if (map1[s[i]] != t[i]) || (map2[t[i]] != s[i]) {
			return false
		}
	}
	return true
}
```

## [Can Place Flowers](https://leetcode.com/problems/can-place-flowers)

A: 计算连续0的个数，如果连续0的个数大于等于n，则可以种花。

```cpp
class Solution {
public:
    bool canPlaceFlowers(vector<int>& flowerbed, int n) {
        int count = 0;
        for (int i = 0; i < flowerbed.size(); i++) {
            if (flowerbed[i] == 0 && (i == 0 || flowerbed[i - 1] == 0) && (i == flowerbed.size() - 1 || flowerbed[i + 1] == 0)) {
                flowerbed[i] = 1;
                count++;
            }
        }
        return count >= n;
    }
};
```

A: 贪心，前后两次遍历，第一次种花第二次拔花。

```go
func canPlaceFlowers(flowerbed []int, n int) bool {
    i := 0
    for i < len(flowerbed) {
        if flowerbed[i] == 1 {
            i += 2
        } else {
            if flowerbed[i] == 0 && (i == 0 || flowerbed[i-1] != 1) {
                n--
                flowerbed[i] = 1
                i += 2
            } else {
                i++
            }
        }
    }
    i = len(flowerbed) - 1
    for i >= 0 {
        if flowerbed[i] == 1 && i != len(flowerbed) - 1 && flowerbed[i + 1] == 1 {
            n++
        }
        i--
    }
    return n <= 0
}
```

## [Majority Element](https://leetcode.com/problems/majority-element)

A: Boyer-Moore 投票算法，如果count为0，则将当前元素赋值给candidate，否则count加1或减1。即，遇到不同数字则count减1，遇到相同数字则count加1，最后剩下的数字就是出现次数最多的数字。

```cpp
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        int count = 0;
        int candidate = 0;
        for (int num : nums) {
            if (count == 0) {
                candidate = num;
            }
            count += (num == candidate) ? 1 : -1;
        }
        return candidate;
    }
};
```

## [Next Greater Element I](https://leetcode.com/problems/next-greater-element-i)

A: map保存nums2中每个元素的下一个更大元素，stack保存nums2中的元素，如果当前元素大于栈顶元素，则栈顶元素的下一个更大元素就是当前元素。

```cpp
class Solution {
public:
    vector<int> nextGreaterElement(vector<int>& nums1, vector<int>& nums2) {
        unordered_map<int, int> m;
        stack<int> s;
        for (int num : nums2) {
            while (!s.empty() && s.top() < num) {
                m[s.top()] = num;
                s.pop();
            }
            s.push(num);
        }
        for (int i = 0; i < nums1.size(); i++) {
            nums1[i] = m.count(nums1[i]) ? m[nums1[i]] : -1;
        }
        return nums1;
    }
};
```

```go
func nextGreaterElement(nums1 []int, nums2 []int) []int {
    ans := make([]int, len(nums1))
    for i := range ans {
        ans[i] = -1
    }
    m := map[int]int{}
    for i, n := range nums1 {
        m[n] = i
    }
    stk := []int{}
    for _, n := range nums2 {
        if len(stk) == 0 {
            stk = append(stk, n)
        } else {
            for len(stk) > 0 && stk[len(stk) - 1] < n {
                back := stk[len(stk) - 1]
                if _, ok := m[back]; ok {
                    ans[m[back]] = n
                }
                stk = stk[:len(stk) - 1]
            }
            stk = append(stk, n)
        }
    }
    return ans
}
```

## [Next Greater Element II](https://leetcode.com/problems/next-greater-element-ii)

A: 遍历两边数组即可。

```go
func nextGreaterElements(nums []int) []int {
    length := len(nums)
    result := make([]int,length,length)
    for i:=0;i<len(result);i++{
        result[i] = -1
    }
    //单调递减，存储数组下标索引
    stack := make([]int,0)
    for i:=0;i<length*2;i++{
        for len(stack)>0&&nums[i%length]>nums[stack[len(stack)-1]]{
            index := stack[len(stack)-1]
            stack = stack[:len(stack)-1] // pop
            result[index] = nums[i%length]
        }
        stack = append(stack,i%length)
    }
    return result
}
```

## [Find Pivot Index](https://leetcode.com/problems/find-pivot-index)

A: 遍历数组，同时比较左右两边的和是否相等。

```cpp
class Solution {
public:
    int pivotIndex(vector<int>& nums) {
        // Initialize rightSum to store the sum of all the numbers strictly to the index's right...
        int rightSum = accumulate(nums.begin(), nums.end(), 0);
        // Initialize leftSum to store the sum of all the numbers strictly to the index's left...
        int leftSum = 0;
        // Traverse all elements through the loop...
        for (int idx = 0; idx < nums.size(); idx++) {
            // subtract current elements with from rightSum...
            rightSum -= nums[idx];
            // If the sum of all the numbers strictly to the left of the index is equal to the sum of all the numbers strictly to the index's right...
            if (leftSum == rightSum)
                return idx;     // Return the pivot index...
            // add current elements with leftSum...
            leftSum += nums[idx];
        }
        return -1;      // If there is no index that satisfies the conditions in the problem statement...
    }
};
```

```go
func pivotIndex(nums []int) int {
	sum := 0
	for _, v := range nums {
		sum += v;
	}

	leftSum := 0    // 中心索引左半和
	rightSum := 0   // 中心索引右半和
	for i := 0; i < len(nums); i++ {
		leftSum += nums[i]
		rightSum = sum - leftSum + nums[i]
		if leftSum == rightSum{
			return i
		}
	}
	return -1
}
```

## [Find All Numbers Disappeared in an Array](https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array)

A: 类似于[Find All Duplicates in an Array](https://leetcode.com/problems/find-all-duplicates-in-an-array)，将数组中的元素作为下标，将对应下标的元素取反，最后遍历数组，如果元素大于0，则说明该下标没有出现过。

```cpp
class Solution {
public:
    vector<int> findDisappearedNumbers(vector<int>& nums) {
        int len = nums.size();
        for(int i=0; i<len; i++) {
            int m = abs(nums[i])-1; // index start from 0
            nums[m] = nums[m]>0 ? -nums[m] : nums[m];
        }
        vector<int> res;
        for(int i = 0; i<len; i++) {
            if(nums[i] > 0) res.push_back(i+1);
        }
        return res;
    }
};
```

## [Maximum Number of Balloons](https://leetcode.com/problems/maximum-number-of-balloons)

A: 计算每个字符出现的次数，然后取balloon要求字符的最小值。

```cpp
class Solution {
public:
    int maxNumberOfBalloons(string text) {
        int count[26] = {0};
        for (char c : text) {
            count[c - 'a']++;
        }
        return min({count['b' - 'a'], count['a' - 'a'], count['l' - 'a'] / 2, count['o' - 'a'] / 2, count['n' - 'a']});
    }
};
```

## [Shuffle the Array](https://leetcode.com/problems/shuffle-the-array)

A: 两个指针（i，2*i+1）分别指向nums的前半部分和后半部分，然后依次取出元素。

```go
func shuffle(nums []int, n int) []int {
    res := make([]int, 2*n)
    for i := 0; i < n; i++ {
        res[2*i] = nums[i]
        res[2*i+1] = nums[i+n]
    }
    return res
}
```

## [Naming a Company](https://leetcode.com/problems/naming-a-company)

A: 用map of set存储{首字母，{后缀\}\}，然后遍历所有的首字母组合，计算出所有的组合数。 

```cpp
class Solution {
public:
    long long distinctNames(vector<string>& ideas){
        unordered_map<int, unordered_set<string>>mpp;
        long long res = 0;

        for(auto &it : ideas) mpp[it[0]].insert(it.substr(1,it.size()-1));        

        for(int i = 0; i < 26; i++) {
            for(int j = i+1; j < 26; j++) {
                unordered_set<string>temp1 = mpp[i+'a'], temp2 = mpp[j+'a'];
                
                long long val = 0;

                for(auto &it : temp1) {
                  if(temp2.find(it) != temp2.end()) val++;
                }
                
                res += 1LL*((temp1.size()- val)*(temp2.size()- val))*2;
            }
        }

        return res;
    }
};
```

## [Spiral Matrix II](https://leetcode.com/problems/spiral-matrix-ii)

A: 以拐点为每个方向填充的起始点，注意奇偶数两种情况和offset的更新。

```cpp
class Solution {
public:
    vector<vector<int>> generateMatrix(int n) {
        int l = 0, t = 0, r = n - 1, b = n - 1;
        vector<vector<int>> ans(n, vector<int>(n, 0));
        int cur = 1;
        while (cur <= n * n) {
            if (t <= b) {
                for (int i = l; i <= r; i++) {
                    ans[t][i] = cur;
                    cur++;
                }
                t++;
            }
            if (l <= r) {
                for (int i = t; i <= b; i++) {
                    ans[i][r] = cur;
                    cur++;
                }
                r--;
            }
            if (t <= b) {
                for (int i = r; i >= l; i--) {
                    ans[b][i] = cur;
                    cur++;
                }
                b--;
            }
            if (l <= r) {
                for (int i = b; i >= t; i--) {
                    ans[i][l] = cur;
                    cur++;
                }
                l++;
            }
        }
        return ans;
    }
};
```

```go
func generateMatrix(n int) [][]int {
    ans := make([][]int, n)
    for idx := range ans {
        ans[idx] = make([]int, n)
    }
    startX, startY := 0, 0
    mid, loop := n / 2, n / 2
    cnt, offset := 1, 1

    for ;loop > 0; loop-- {
        i, j := startX, startY
        for j = startY; j < startY + n - offset; j++ {
            ans[startX][j] = cnt
            cnt++
        }
        for i = startX; i < startX + n - offset; i++ {
            ans[i][j] = cnt
            cnt++
        }
        for ; j > startY; j-- {
            ans[i][j] = cnt
            cnt++
        }
        for ; i > startX; i-- {
            ans[i][j] = cnt
            cnt++
        }
        startX++
        startY++
        offset += 2
    }
    
    if n % 2 != 0 {
        ans[mid][mid] = cnt
    }
    return ans
}
```

## [Intersection of Two Arrays](https://leetcode.com/problems/intersection-of-two-arrays)

A: 用哈希表记录nums1，遍历nums2，出现相同元素则加入ans，同时**删除**相应的哈希表元素。

```go
func intersection(nums1 []int, nums2 []int) []int {
    s := make(map[int]struct{})
    for k := range nums1 {
        s[nums1[k]] = struct{}{}
    }
    ans := []int{}
    for k := range nums2 {
        if _, ok := s[nums2[k]]; ok {
            ans = append(ans, nums2[k])
            delete(s, nums2[k])
        }
    }
    return ans
}
```

## [Happy Number](https://leetcode.com/problems/happy-number)

A: 哈希表记录中间计算结果，若重复则不是快乐数。

```go
func isHappy(n int) bool {
    s := make(map[int]struct{})
    for n != 1 {
        cur := 0
        for n != 0 {
            digit := n % 10
            n /= 10
            cur += digit * digit
        }
        if _, ok := s[cur]; ok {
            return false
        }
        s[cur] = struct{}{}
        n = cur
    }
    return true
}
```

## [4Sum II](https://leetcode.com/problems/4sum-ii)

A: 分为两个部分，计算前两个数组和的情况，记录{sum, count}，在后面两个数组的求和中查询对应的相反数。

```go
func fourSumCount(nums1 []int, nums2 []int, nums3 []int, nums4 []int) int {
    s := make(map[int]int)
    ans := 0
    for i := 0; i < len(nums1); i++ {
        for j := 0; j < len(nums2); j++ {
            s[nums1[i] + nums2[j]]++
        }
    }
    for i := 0; i < len(nums3); i++ {
        for j := 0; j < len(nums4); j++ {
            if _, ok := s[-(nums3[i] + nums4[j])]; ok {
                ans += s[-(nums3[i] + nums4[j])]
            }
        }
    }
    return ans
}
```

## [Ransom Note](https://leetcode.com/problems/ransom-note)

A: 记录字符出现次数，在另一个字符串中减去，如果出现异常值则无法construct。

```go
func canConstruct(ransomNote string, magazine string) bool {
    s := make(map[rune]int)
    for _, m := range magazine {
        s[m]--
    }
    for _, rn := range ransomNote {
        s[rn]++
        if s[rn] > 0 {
            return false
        }
    }
    return true
}
```

A: 库函数。

`strings.Count`: 统计字符串中某个字符出现的次数。

```go
func canConstruct(ransomNote string, magazine string) bool {
    	for _, v := range ransomNote {
		if strings.Count(ransomNote, string(v)) > strings.Count(magazine, string(v)) {
			return false
		}
	}
	return true
}
```

## [Add to Array-Form of Integer](https://leetcode.com/problems/add-to-array-form-of-integer)

A: 类似于[Add Binary](https://leetcode.com/problems/add-binary)。

```go
func addToArrayForm(A []int, K int) []int {
    ans := []int{}
    carry := 0
    for i := len(A) - 1; i >= 0; i-- {
        sum := A[i] + K % 10 + carry
        carry = sum / 10
        ans = append(ans, sum % 10)
        K /= 10
    }
    for K != 0 {
        sum := K % 10 + carry
        carry = sum / 10
        ans = append(ans, sum % 10)
        K /= 10
    }
    if carry != 0 {
        ans = append(ans, carry)
    }
    for i := 0; i < len(ans) / 2; i++ {
        ans[i], ans[len(ans) - 1 - i] = ans[len(ans) - 1 - i], ans[i]
    }
    return ans
}
```

## [Range Sum Query - Immutable](https://leetcode.com/problems/range-sum-query-immutable)

A: 前缀和。

```cpp
class NumArray {
private:
    vector<int> prefix;
public:
    NumArray(vector<int>& nums) {
        prefix = nums;
        for (int i = 1; i < nums.size(); i++) {
            prefix[i] += prefix[i - 1];
        }
    }
    
    int sumRange(int left, int right) {
        if (left == 0) {
            return this->prefix[right];
        } else {
            return this->prefix[right] - this->prefix[left - 1];
        }
    }
};

/**
 * Your NumArray object will be instantiated and called as such:
 * NumArray* obj = new NumArray(nums);
 * int param_1 = obj->sumRange(left,right);
 */
```

```go
type NumArray struct {
    prefix []int
}


func Constructor(nums []int) NumArray {
    prefix := make([]int, 0, len(nums))
    sum := 0
    for _, v := range nums {
        sum += v
        prefix = append(prefix, sum)
    }
    return NumArray{prefix: prefix}
}


func (this *NumArray) SumRange(left int, right int) int {
    if left == 0 {
        return this.prefix[right]
    } else {
        return this.prefix[right] - this.prefix[left - 1]
    }
}


/**
 * Your NumArray object will be instantiated and called as such:
 * obj := Constructor(nums);
 * param_1 := obj.SumRange(left,right);
 */
 ```

 ## [Design HashMap](https://leetcode.com/problems/design-hashmap)

 A: 当出现哈希碰撞时，用链表扩容。

 ```go
 type Node struct {
    key int
    value int
    next *Node
}

type MyHashMap struct {
    items []*Node
    size int
}


/** Initialize your data structure here. */
func Constructor() MyHashMap {
    size := 1000
    return MyHashMap {
        make([]*Node, size),
        size,
    }
}


/** value will always be non-negative. */
func (this *MyHashMap) Put(key int, value int)  {
    index := key % this.size
    if this.items[index] == nil {
        this.items[index] = &Node{key: key, value: value}
        return
    }
    
    item := this.items[index]
    
    for item != nil {
        if item.key == key {
            item.value = value
            return
        }
        
        if item.next == nil {
            item.next = &Node{key: key, value: value}
            return
        }
        
        item = item.next
    }
}


/** Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key */
func (this *MyHashMap) Get(key int) int {
    index := key % this.size
    if this.items[index] == nil {
        return -1
    }
    
    item := this.items[index]
    
    for item != nil {
        if item.key == key {
            return item.value
        }
        
        item = item.next
    }
    
    return -1
}


/** Removes the mapping of the specified value key if this map contains a mapping for the key */
func (this *MyHashMap) Remove(key int)  {
    index := key % this.size
    if this.items[index] == nil {
        return
    }
    
    item := this.items[index]
    
    if item.key == key {
        this.items[index] = item.next
        return
    }
    
    prev, item := item, item.next
    
    for item != nil {
        if item.key == key {
            prev.next = item.next
            return
        }
        
        prev = item        
        item = item.next
    }
}


/**
 * Your MyHashMap object will be instantiated and called as such:
 * obj := Constructor();
 * obj.Put(key,value);
 * param_2 := obj.Get(key);
 * obj.Remove(key);
 */
 ```

 ## [Sort an Array](https://leetcode.com/problems/sort-an-array)

 A: 归并排序。

```go
func sortArray(nums []int) []int {
    l, r := 0, len(nums)-1
    return mergesort(nums, l, r) 
}

func mergesort(nums []int, l, r int) []int {
    if l == r {
        return []int{nums[l]}
    }
    mid := l + (r-l)/2
    return merge(mergesort(nums, l, mid), mergesort(nums, mid+1, r))
}

func merge(left, right []int) []int {
    m, n := len(left), len(right)
    res := make([]int, m+n, m+n)
    i, j := 0, 0
    for k:=0; k<m+n; k++ {
        if i == m {
            res[k] = right[j]
            j += 1
        } else if j == n {
            res[k] = left[i]
            i += 1
        } else if left[i] < right[j] {
            res[k] = left[i]
            i += 1
        } else {
            res[k] = right[j]
            j += 1
        }
    }
    return res
}
```

## [Split the Array to Make Coprime Products](https://leetcode.com/problems/split-the-array-to-make-coprime-products)

A: 不需要计算所有乘积，也不需要用gcd函数判断是否互质，只需要计算每个数的质因数，判断某一个下标位置前后是否存在公共质因数即可。

```cpp
vector<int> factorize(int n) {
    vector<int> res;
    for (int i = 2; n > 1 && i < 1000; i += 1 + (i % 2))
        if (n % i == 0) {
            res.push_back(i);
            while(n % i == 0)
                n /= i;
        }
    if (n > 1)
        res.push_back(n);
    return res;
}
int findValidSplit(vector<int>& nums) {
    unordered_map<int, int> left, right;
    for (int n : nums)
        for (int f : factorize(n))
            ++right[f];
    for (int i  = 0, common = 0; i < nums.size() - 1; ++i) {
        for (int f : factorize(nums[i]))
            // left[f]++ == 0 means f is not in left
            // left[f] == right[f] means f is in both left and right
            common += (left[f]++ == 0) - (left[f] == right[f]);
        if (common == 0)
            return i;        
    }
    return -1;
}
```

## [Count the Number of Vowel Strings in Range](https://leetcode.com/problems/count-the-number-of-vowel-strings-in-range)

A: 在给定范围内判断元素首尾字母。

```go
func vowelStrings(words []string, left int, right int) int {
    set := map[byte]bool{'a': true,'e': true,'i': true,'o': true,'u': true}
    ans := 0
    for i := left; i <= right; i++ {
        if set[words[i][0]] == true && set[words[i][len(words[i])-1]] == true {
            ans++
        }
    }
    return ans
}
```

## [Concatenation of Array](https://leetcode.com/problems/concatenation-of-array)

A: 遍历，按顺序放入元素。

```go
func getConcatenation(nums []int) []int {
    ans := make([]int, 2 * len(nums))
    for i := 0; i < len(nums); i++ {
        ans[i] = nums[i]
        ans[i + len(nums)] = nums[i]
    }
    return ans
}
```

## [How Many Numbers Are Smaller Than the Current Number](https://leetcode.com/problems/how-many-numbers-are-smaller-than-the-current-number)

A: sort以得到小于该数的个数，map存储。

```go
func smallerNumbersThanCurrent(nums []int) []int {
    // map，key[数组中出现的数] value[比这个数小的个数]
    m := make(map[int]int)
    // 拷贝一份原始数组
    rawNums := make([]int,len(nums)) 
    copy(rawNums,nums)
    // 将数组排序
    sort.Ints(nums)
    // 循环遍历排序后的数组，值为map的key，索引为value
    for i,v := range nums {
        _,contains := m[v]
        if !contains {
          m[v] = i
        } 
          
    }
    // 返回值结果
    result := make([]int,len(nums))
    // 根据原始数组的位置，存放对应的比它小的数
    for i,v := range rawNums {
        result[i] = m[v]
    }

    return result
}
```

## [Valid Mountain Array](https://leetcode.com/problems/valid-mountain-array)

A: 遍历，判断上升和下降。

```go
func validMountainArray(arr []int) bool {
	if len(arr) < 3 {
		return false
	}

	i := 1
	flagIncrease := false // 上升
	flagDecrease := false // 下降

	for ; i < len(arr) && arr[i-1] < arr[i]; i++ {
		flagIncrease = true;
	}

	for ; i < len(arr) && arr[i-1] > arr[i]; i++ {
		flagDecrease = true;
	}

	return i == len(arr) && flagIncrease && flagDecrease;
}
```

## [Unique Number of Occurrences](https://leetcode.com/problems/unique-number-of-occurrences)

A: 遍历，用map存储次数，用set判断是否重复。

```go
func uniqueOccurrences(arr []int) bool {
    m := map[int]int{}
    s := map[int]struct{}{}
    for _, n := range arr {
        m[n]++
    }
    for _, c := range m {
        if _, ok := s[c]; ok {
            return false
        } else {
            s[c] = struct{}{}
        }
    }
    return true
}
```

## [Rotate Array](https://leetcode.com/problems/rotate-array)

A: 使用类似于插入排序的方法，每次将最后一个元素放到第一个位置。

```go
func rotate(nums []int, k int)  {
    for i := 0; i < k; i++ {
        tmp := nums[len(nums) - 1]
        copy(nums[1:], nums[:len(nums) - 1])
        nums[0] = tmp
    }
}
```

A: 反转字符。

```go
func rotate(nums []int, k int)  {
    l:=len(nums)
    index:=l-k%l
    reverse(nums)
    reverse(nums[:l-index])
    reverse(nums[l-index:])
}

func reverse(nums []int){
    l:=len(nums)
    for i:=0;i<l/2;i++{
        nums[i],nums[l-1-i]=nums[l-1-i],nums[i]
    }
}
```

## [Sort Array By Parity II](https://leetcode.com/problems/sort-array-by-parity-ii)

A: 每次找到奇偶不匹配的元素，交换。

```go
func sortArrayByParityII(nums []int) []int {
    odd := 1
    for i := 0; i < len(nums); i += 2 {
        if nums[i] % 2 != 0 {
            for nums[odd] % 2 != 0 {
                odd += 2
            }
            nums[i], nums[odd] = nums[odd], nums[i]
        }
    }
    return nums
}
```

## [Minimize Maximum of Array](https://leetcode.com/problems/minimize-maximum-of-array)

A: [详解](https://leetcode.com/problems/minimize-maximum-of-array/solutions/2706521/java-c-python-prefix-sum-average-o-n/)

```go
func minimizeArrayValue(nums []int) int {
    sum, ans := 0, 0
    for i, n := range nums {
        sum += n
        ans = max(ans, (sum + i) / (i + 1))
    }
    return ans
}

func max(i, j int) int {
    if i > j {
        return i
    } else {
        return j
    }
}
```

## [Find Common Characters](https://leetcode.com/problems/find-common-characters)

A: 对每个字符串进行词频统计，然后查找每一列的最小值。

```go
func commonChars(words []string) []string {
    length:=len(words)
    fre:=make([][]int,0)//统计每个字符串的词频
    res:=make([]string,0)
    //统计词频
    for i:=0;i<length;i++{
        var row [26]int//存放该字符串的词频
        for j:=0;j<len(words[i]);j++{
            row[words[i][j]-97]++
        }
        fre=append(fre,row[:])
    }
    //查找一列的最小值
    for j:=0;j<len(fre[0]);j++{
        pre:=fre[0][j]
        for i:=0;i<len(fre);i++{
            pre=min(pre,fre[i][j])
        }
        //将该字符添加到结果集（按照次数）
        tmpString:=string(j+97)
        for i:=0;i<pre;i++{
            res=append(res,tmpString)
        }
    }
    return res
}

func min(a,b int)int{
    if a>b{
        return b
    }
    return a
}
```

## [Long Pressed Name](https://leetcode.com/problems/long-pressed-name)

A: 同时遍历两个字符串，如果两个字符相等，那么继续遍历，如果不相等，那么判断typed的当前字符是否和上一个字符相等，如果相等，那么继续遍历，如果不相等，那么返回false。

```go
func isLongPressedName(name string, typed string) bool {
	if(name[0] != typed[0] || len(name) > len(typed)) {
		return false;
	}

	idx := 0 // name的索引
	var last byte  // 上个匹配字符
	for i := 0; i < len(typed); i++ { // typed的索引
		if idx < len(name) && name[idx] == typed[i] {
			last = name[idx]
			idx++
		} else if last == typed[i] {
            // long pressed
			continue
		} else  {
			return false
		}
	}
	return idx == len(name)
}
```

## [Next Permutation](https://leetcode.com/problems/next-permutation)

A: 从后往前遍历，找到第一组顺序元素，然后将这两个元素交换，然后将后面的逆序元素反转。

```go
func nextPermutation(nums []int)  {
    for i:=len(nums)-1;i>=0;i--{
        for j:=len(nums)-1;j>i;j--{
            if nums[j]>nums[i]{
                nums[j], nums[i] = nums[i], nums[j]
                reverse(nums,0+i+1,len(nums)-1)
                return 
            }
        }
    }
    reverse(nums,0,len(nums)-1)
}

func reverse(a []int,begin,end int){
    for i,j:=begin,end;i<j;i,j=i+1,j-1{
        a[i],a[j]=a[j],a[i]
    }
}
```

## [Range Addition](https://www.lintcode.com/problem/903/)

A: 差分数组。

```go
/**
 * @param length: the length of the array
 * @param updates: update operations
 * @return: the modified array after all k operations were executed
 */
func GetModifiedArray(length int, updates [][]int) []int {
    diff := make([]int, length)
    for i := range updates {
        diff[updates[i][0]] += updates[i][2]
        if updates[i][1] + 1 < length {
            diff[updates[i][1] + 1] -= updates[i][2]
        }
    }
    for i := 1; i < length; i++ {
        diff[i] += diff[i - 1]
    }
    return diff
}
```

## [Corporate Flight Bookings](https://leetcode.com/problems/corporate-flight-bookings)

A: 差分数组。

```go
func corpFlightBookings(bookings [][]int, n int) []int {
    ans := make([]int, n)
    for i := range bookings {
        ans[bookings[i][0] - 1] += bookings[i][2]
        if bookings[i][1] < n {
            ans[bookings[i][1]] -= bookings[i][2]
        }
    }
    for i := 1; i < n; i++ {
        ans[i] += ans[i - 1]
    }
    return ans
}
```

## [Random Pick with Weight](https://leetcode.com/problems/random-pick-with-weight)

A: 前缀和+二分查找。[详解](https://labuladong.github.io/algo/di-yi-zhan-da78c/shou-ba-sh-48c1d/dai-quan-z-585d6/)

```go
type Solution struct {
    prefix []int
}


func Constructor(w []int) Solution {
    for i := 1; i < len(w); i++ {
        w[i] += w[i - 1]
    }
    return Solution{prefix: w}
}


func (this *Solution) PickIndex() int {
    w := this.prefix
    target := rand.Intn(w[len(w) - 1]) + 1
    l, r := 0, len(w) - 1
    for l <= r {
        mid := l + (r - l) / 2
        if w[mid] == target {
            return mid
        } else if w[mid] > target {
            r--
        } else {
            l++
        }
    }
    return l
}


/**
 * Your Solution object will be instantiated and called as such:
 * obj := Constructor(w);
 * param_1 := obj.PickIndex();
 */
```
