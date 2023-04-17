# Sliding Window

+ **滑动窗口**：用窗口遍历线性表，指针初始位于线性表同侧。
+ **双指针**：指针初始位于线性表两端。

## [Best Time to Buy And Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock)

A: 滑动窗口，向右滑动sell，用最小值更新buy。

```cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int ans = 0;
        int buy = 0, sell = 1;
        int len = prices.size();
        while (sell < len) {
            int p = prices[sell] - prices[buy];
            if (p > 0)  ans = max(ans, p);
            else    buy = sell;
            sell++;
        }
        return ans;
    }
};
```

```go
func maxProfit(prices []int) int {
    if len(prices) <= 1 {
        return 0
    } 
    
    min, maxSale := prices[0], 0

    for _,price := range prices {
        if price < min {
            min = price
        } else if (price-min) > maxSale{
            maxSale = price - min
        }
    }
    return  maxSale
}
```

## [Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters)

A: 集合存放字符，检验集合中是否存在字符，以此移动左指针。

```cpp
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_set<char> letters;
        
        int i = 0;
        int j = 0;
        
        int result = 0;
        
        while (j < s.size()) {
            if (letters.find(s[j]) == letters.end()) {
                letters.insert(s[j]);
                result = max(result, j - i + 1);
                j++;
            } else {
                letters.erase(s[i]);
                i++;
            }
        }
        
        return result;
    }
};
```

## [Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement)

A: 判断次小字符集之和与k的大小，以此滑动左指针。

```cpp
class Solution {
public:
    int characterReplacement(string s, int k) {
        vector<int> count(26);
        int maxCount = 0;
        
        int i = 0;
        int j = 0;
        
        int result = 0;
        
        while (j < s.size()) {
            count[s[j] - 'A']++;
            maxCount = max(maxCount, count[s[j] - 'A']);
            if (j - i + 1 - maxCount > k) {
                count[s[i] - 'A']--;
                i++;
            }
            result = max(result, j - i + 1);
            j++;
        }
        
        return result;
    }
};
```

## [Minumum Window Substring](https://leetcode.com/problems/minimum-window-substring)

A: remaining标记ASCII中所有**substr**还未包含的字符次数，require标记t中所有要求包含的字符数。

128: ASCII表的大小。

```cpp
class Solution {
public:
    string minWindow(string s, string t) {
        if (s.size() == 0 || t.size() == 0) return "";
        vector<int> remaining(128, 0);
        int required = t.size();
        for (int i = 0; i < required; i++) remaining[t[i]]++;
        // left is the start index of the min-length substring ever found
        // start、end为窗口边界
        int min_len = INT_MAX, start = 0, left = 0, end = 0;
        while(end <= s.size() && start < s.size()) {
            if(required) {
                // 现有substr不满足t，右指针右移，直至满足要求（require == 0）
                if (end == s.size()) break;
                remaining[s[end]]--;
                // 若remaining中该字符仍为非负，则一定是t的字符，说明窗口包含t中字符，require减小
                if (remaining[s[end]] >= 0) required--;
                end++;
            } else {
                // 现有substr满足t，左指针右移，找到最短substr
                if (end - start < min_len) {
                    min_len = end -start;
                    left = start;
                }
                remaining[s[start]]++;
                // 若remianing中该字符为正，则一定是t中字符，说明窗口排除了一个t中字符，require增加
                if (remaining[s[start]] > 0) required++;
                start++;
            }
        }
        return min_len == INT_MAX? "" : s.substr(left, min_len);
    }
};
```

## [Permutation In String](https://leetcode.com/problems/permutation-in-string)

A: 滑动窗口，用两个数组存储字母状态。

```cpp
class Solution {
public:
    bool checkInclusion(string s1, string s2) {
        if (s1.size() > s2.size()) return false;
        vector<int> s1v (26, 0);
        for (auto c : s1) s1v[c - 'a']++;
        vector<int> s2v (26, 0);
        int l = 0, r = 0;
        while (r < s2.size()) {
            s2v[s2[r]-'a']++;
            if (r - l + 1 == s1.size()) 
                if (s1v == s2v) return true; // 直接判断vector是否相等
            if (r - l + 1 < s1.size()) r++;
            else {
                s2v[s2[l]-'a']--;
                l++;
                r++;
            }
        }
        return false;
    }
};
```

```go
func checkInclusion(s1 string, s2 string) bool {
    m1 := map[byte]int{}
    m2 := map[byte]int{}
    for i := range s1 {
        m1[s1[i]]++
    }
    l, r, valid := 0, 0, 0
    for r < len(s2) {
        if _, ok := m1[s2[r]]; ok {
            m2[s2[r]]++
            if m1[s2[r]] == m2[s2[r]] {
                valid++
            }
        }
        r++
        for r - l >= len(s1) {
            if valid == len(m1) {
                return true
            }
            if _, ok := m1[s2[l]]; ok {
                if m1[s2[l]] == m2[s2[l]] {
                    valid--
                }
                m2[s2[l]]--
            }
            l++
        }
    }
    return false
}
```

## [Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum)

A: **单调队列**，维持队列的降序，进入队列时剔除无需再比较的更小元素。

```cpp
class Solution {
public:
   vector<int> maxSlidingWindow(vector<int>& nums, int k) {
       deque<int> queue;
       vector<int> ans;
       for (int left = 0; left < nums.size(); ++left) {
           // As the window move on, element nums[left-k] will be outdated.
           if (queue.front() == left - k) queue.pop_front();
           // Now we are ready to push our new element nums[left]'s index into the queue.
           // But before that, we should clear elements which is smaller then nums[left].
           // Why? Because if nums[left] is bigger then nums[i], 
           // there will be no way for nums[i] be selected as the max number in range (left-k, left]
           while (!queue.empty() && nums[queue.back()] < nums[left]) queue.pop_back();
           // Now push the index into our queue.
           queue.push_back(left);
           // Okay, now nums[queue.front()] mush be the max number in range (left-k, left] 
           if (left - k + 1 >= 0) ans.push_back(nums[queue.front()]);
       }
       return ans;
   }
};
```

```go
func maxSlidingWindow(nums []int, k int) []int {
    ans := []int{}
    q := []int{}
    for i := 0; i < len(nums); i++ {
        if len(q) == 0 {
            q = append(q, i)
        } else {
            if nums[i] < nums[q[len(q) - 1]] {
                q = append(q, i)
            } else {
                for len(q) > 0 && nums[q[len(q) - 1]] < nums[i] {
                    q = q[:len(q) - 1]
                }
                q = append(q, i)
            }
        }
        if i >= k - 1 {
            if q[0] < i - k + 1 {
                q = q[1:]
            }
            ans = append(ans, nums[q[0]])
        }
    }
    return ans
}
```

## [Frequency of The Most Frequent Element](https://leetcode.com/problems/frequency-of-the-most-frequent-element)

A: 排序，动态滑动窗口，需要达到窗口右侧最大值。

```java
public int maxFrequency(int[] A, int k) {
    int res = 1, left = 0;
    long sumOfElementsInWindow = 0;
    Arrays.sort(A);
    for (int right = 0; right < A.length; ++right) {
        sumOfElementsInWindow += A[right];
        // increase the left pointer until the condition satisfies
        while (getNumberOfOperationsNeeded(left, right, sumOfElementsInWindow, A) > k) {
            sumOfElementsInWindow -= A[left];
            left += 1;
        }
        res = Math.max(res, right - left + 1); // update the window if its the max window
    }
    return res;
}
/**
 Number of operations needed for all elements in the window [startIndex, endIndex] to hit A[endIndex]
 Example:
 Consider arr with [1, 2, 3, 4] with startIndex = 0; endIndex = 3: i.e If 1, 2, 3 wants to become 4.
 Number of operations needed
 = (4-1)+(4-2)+(4-3)+(4-4) = 6.
 =  4 + 4 + 4 + 4 - (1 + 2 + 3+ 4)
 = 4 * 4 - (1 + 2 + 3 + 4)
 = (number of elements) * ElementToReach - sum of elements in the window
 */
long getNumberOfOperationsNeeded(int startIndex, int endIndex, long sumOfElementsInWindow, int[] A){
    int numberOfElements = endIndex - startIndex + 1;
    int elementToReach = A[endIndex];
    return ((long) numberOfElements * elementToReach) - sumOfElementsInWindow;
}
```

## [Minimum Number of Flips to Make The Binary String Alternating](https://leetcode.com/problems/minimum-number-of-flips-to-make-the-binary-string-alternating)

A: 最终生成两种情况：1在奇数或0在奇数。计算两种情况的操作2数目，滑动窗口模拟操作1。

```cpp
class Solution {
public:
    int minFlips(string s) {
        int n = s.size();
        s += s; // 操作1
        string s1, s2;
        
        for(int i = 0; i < s.size(); i++) {
            s1 += i % 2 ? '0' : '1';
            s2 += i % 2 ? '1' : '0';
        }
        int ans1 = 0, ans2 = 0, ans = INT_MAX;
        for(int i = 0; i < s.size(); i++) {
            if(s1[i] != s[i]) ++ans1;
            if(s2[i] != s[i]) ++ans2;
            if(i >= n) { //the most left element is outside of sliding window, we need to subtract the ans if we did `flip` before.
            // 模拟操作1，之前用操作2得到的结果可被抵消
                if(s1[i - n] != s[i - n]) --ans1;
                if(s2[i - n] != s[i - n]) --ans2;
            }
            if(i >= n - 1)
                ans = min({ans1, ans2, ans});
        }
        return ans;
    }
};

class Solution {
public:
    int minFlips(string s) {
        int n = s.size();
        int ans1 = 0, ans2 = 0, ans = INT_MAX;
        for(int i = 0; i < 2 * n; i++) {
            if(i < n) s[i] -= '0'; //make '1' and '0' to be integer 1 and 0.
            if(i % 2 != s[i % n]) ++ans1;
            if((i + 1) % 2 != s[i % n]) ++ans2;
            if(i >= n) {
                if((i - n) % 2 != s[i - n]) --ans1;
                if((i - n + 1) % 2 != s[i - n]) --ans2;
            }
            if(i >= n - 1)
                ans = min({ans1, ans2, ans});
        }
        return ans;
    }
};
```

## [Maximum Points You Can Obtain From Cards](https://leetcode.com/problems/maximum-points-you-can-obtain-from-cards)

A: 找到总和最小的连续子数组。

```cpp
class Solution {
public:
    int maxScore(vector<int>& cardPoints, int k) {
        int res = 0;
        //First k elements in our window
        for(int i=0;i<k;i++) res+=cardPoints[i];
        
        int curr=res;
        for(int i=k-1;i>=0;i--) {
            //We remove the last visited element and add the non-visited element from the last
            curr-=cardPoints[i];
            curr+=cardPoints[cardPoints.size()-k+i];
            //We check the maximum value any possible combination can give
            res = max(res, curr);
        }
        
        return res;
    }
};
```

## [Minimum Size Subarray Sum](https://leetcode.com/problems/minimum-size-subarray-sum)

A: 滑动窗口确定最小值。

```cpp
class Solution {
public:
    int minSubArrayLen(int s, vector<int>& nums) {
        int l = 0, r = 0, n = nums.size(), sum = 0, len = INT_MAX;
        while (r < n) {
            sum += nums[r++];
            while (sum >= s) {
                len = min(len, r - l);
                sum -= nums[l++];
            }
        }
        return len == INT_MAX ? 0 : len;
    }
};
```

```go
func minSubArrayLen(target int, nums []int) int {
    ans, i, j, sum:= math.MaxInt, 0, 0, 0
    for j < len(nums) {
        sum += nums[j]
        for sum >= target {
            ans = min(ans, j - i + 1)
            sum -= nums[i]
            i++
        }
        j++
    }
    if ans == math.MaxInt {
        ans = 0
    }
    return ans
}

func min(n1, n2 int) int {
    if n1 < n2 {
        return n1
    } else {
        return n2
    }
}
```

## [Find K Closest Elements](https://leetcode.com/problems/find-k-closest-elements)

A: 利用二分查找找到最接近x的元素，双指针法。

```cpp
class Solution {
public:
    vector<int> findClosestElements(vector<int>& arr, int k, int x) {
        vector<int> ans;
        if (arr[0] >= x) {
            return vector<int> (arr.begin(), arr.begin() + k);
        }
        if (arr[arr.size() - 1] <= x) {
            return vector<int> (arr.end() - k, arr.end());
        }
        auto low = lower_bound(arr.begin(), arr.end(), x);
        if (low != arr.begin()) {
            low = abs(*low - x) < abs(*(low - 1) - x) ? low : low - 1;
        }
        auto start = low - 1, end = low + 1;
        while (end - start - 1 < k) {
            if (start < arr.begin()) {
                end = arr.begin() + k;
                break;
            } else if (end == arr.end()) {
                start = arr.end() - 1 - k;
                break;
            }
            if (abs(*start - x) <= abs(*end - x)) {
                start--;
            } else {
                end++;
            }
        }
        for (auto i = start + 1; i != end; i++) {
            ans.push_back(*i);
        }
        return ans;
    }
};
```

## [Fruit Into Baskets](https://leetcode.com/problems/fruit-into-baskets)

A: 滑动窗口。

```go
func totalFruit(tree []int) int {
    var res int
    var m = make(map[int]int)
    for i, j := 0, 0; j < len(tree); j++ {
        m[tree[j]]++
        for len(m) > 2 {
            m[tree[i]]--
            if m[tree[i]] == 0 {
                delete(m, tree[i])
            }
            i++
        }
        res = max(res, j-i+1)
    }
    return res
}
```

## [Contains Duplicate II](https://leetcode.com/problems/contains-duplicate-ii)

A: 滑动窗口，哈希表存储窗口元素。

```go
func containsNearbyDuplicate(nums []int, k int) bool {
    m := map[int]struct{}{}
    for i := 0; i < len(nums) && i < k + 1; i++ {
        if _, ok := m[nums[i]]; ok {
            return true
        } else {
            m[nums[i]] = struct{}{}
        }
    }
    i, j := 0, k
    for j < len(nums) {
        delete(m, nums[i])
        i++
        j++
        if j < len(nums) {
            if _, ok := m[nums[j]]; ok {
                return true
            } else {
                m[nums[j]] = struct{}{}
            }
        }
    }
    return false
}
```

## [Number of Sub-array of Size K and Average Greater than or Equal to Threshold](https://leetcode.com/problems/number-of-sub-arrays-of-size-k-and-average-greater-than-or-equal-to-threshold)

A: 滑动窗口。

```go
func numOfSubarrays(arr []int, k int, threshold int) int {
    left,sum,count := 0,0,0
	for right :=0; right < len(arr); right++ {
		sum += arr[right]
		if( right - left + 1 == k) {
			if  sum / k >= threshold {
				count += 1
			}
			sum -= arr[left]
			left +=1
		}
	}
	return count
}
```

## [Longest Turbulent Subarray](https://leetcode.com/problems/longest-turbulent-subarray)

A: 滑动窗口，注意等号时R的移动。

```go
func maxTurbulenceSize(arr []int) int {
    if len(arr) <= 1 {
        return len(arr)
    }
    res, left, right, lastNum, flag := 0, 0, 0, arr[0], arr[1]-arr[0]
    for left < len(arr) {
        if right < len(arr)-1 && ((arr[right+1] > lastNum && flag > 0) || (arr[right+1] < lastNum && flag < 0) || (right == left)) {
            right++
            flag = lastNum-arr[right]
            lastNum = arr[right]
        } else {
            if left != right && flag != 0 {
                res = max(res, right-left+1)
            }
            left++
        }
    }
    return max(res, 1)
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

## [Number of Zero-Filled Subarrays](https://leetcode.com/problems/number-of-zero-filled-subarrays)

A: 滑动窗口，连续0的数组个数存在数学关系。

```go
func zeroFilledSubarray(nums []int) int64 {
    var ans int64
    i, j := 0, 0
    for j < len(nums) {
        if nums[j] == 0 {
            ans += int64(j - i + 1)
            j++
        } else {
            i = j
            for i < len(nums) && nums[i] != 0 {
                i++
            }
            j = i
        }
    }
    return ans
}
```

## [Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring)

A: 滑动窗口，记录s和t中对应字符出现个数以及valid值，不满足valid则右移j，满足则左移i。

```cpp
class Solution {
public:
    string minWindow(string s, string t) {
        // count of char in t
        unordered_map<char, int> m;
        for (auto c: t) m[c]++;
        
        int i = 0;
        int j = 0;
        
        // # of chars in t that must be in s
        int counter = t.size();
        
        int minStart = 0;
        int minLength = INT_MAX;
        
        while (j < s.size()) {
            // if char in s exists in t, decrease
            if (m[s[j]] > 0) {
                counter--;
            }
            // if char doesn't exist in t, will be -'ve
            m[s[j]]--;
            // move j to find valid window
            j++;
            
            // when window found, move i to find smaller
            while (counter == 0) {
                if (j - i < minLength) {
                    minStart = i;
                    minLength = j - i;
                }
                
                m[s[i]]++;
                // when char exists in t, increase
                if (m[s[i]] > 0) {
                    counter++;
                }
                i++;
            }
        }
        
        if (minLength != INT_MAX) {
            return s.substr(minStart, minLength);
        }
        return "";
    }
};
```

```go
func minWindow(s string, t string) string {
    l, r, valid := 0, 0, 0
    m1 := map[byte]int{}
    m2 := map[byte]int{}
    ans := ""
    for i := range t {
        m1[t[i]]++
    }
    for r < len(s) {
        if _, ok := m1[s[r]]; ok {
            m2[s[r]]++
            if m2[s[r]] == m1[s[r]] {
                valid++
            }
        }
        r++
        for valid == len(m1) {
            if ans == "" || len(ans) > r - l {
                ans = s[l:r]
            }
            if _, ok := m1[s[l]]; ok {
                m2[s[l]]--
                if m2[s[l]] < m1[s[l]] {
                    valid--
                }
            }
            l++
        }
    }
    return ans
}
```
