# Greedy

## [Maximum Subarray](https://leetcode.com/problems/maximum-subarray)

A: 用`cur`存放当前最大和，作为前缀，如果是负数，遍历时舍去，重新开始计算。

```cpp
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int curr = nums[0];
        int result = nums[0];
        
        for (int i = 1; i < nums.size(); i++) {
            curr = max(curr + nums[i], nums[i]);
            result = max(result, curr);
        }
        
        return result;
    }
};
```

A: 滑动窗口。

```go
func maxSubArray(nums []int) int {
    i, j := 0, 1
    ans := nums[i]
    cur := nums[i]
    for j < len(nums) {
        if nums[i] < 0 {
            i++
            cur -= nums[i]
        }
        if cur + nums[j] < 0 {
            i = j
            cur = nums[j]
        } else {
            cur += nums[j]
        }
        if cur > ans {
            ans = cur
        }
        j++
    }
    return ans
}
```

## [Jump Game](https://leetcode.com/problems/jump-game)

A: DP，`nums[i]`是否可达取决于是否存在`nums[j]`可达`nums[i]`。

```cpp
class Solution {
public:
    bool canJump(vector<int>& nums) {
        int len = nums.size();
        vector<bool> dp(len, false);
        dp[0] = true;

        for (int i = 1; i < len; i++) {
            for (int j  = i - 1; j >= 0; j--) {
                if (dp[j] && nums[j] >= i - j) {
                    dp[i] = true;
                    break;
                }
            }
        }

        return dp[len - 1];
    }
};
```

A: Greedy，每次取能到达的最远位置作为局部最优解。

```cpp
bool canJump(vector<int>& nums) {
  int n = nums.size(), farest = 0;
  for(int i = 0;i < n; i++)
  {
    if(farest < i) return false;
    farest = max(i + nums[i], farest);
  }
  
  return true;
}
```

```go
func canJump(nums []int) bool {
    if len(nums) == 1 {
        return true
    }
    cover := 0
    for i := 0; i <= cover; i++ {
        if nums[i] + i > cover {
            cover = nums[i] + i
        }
        if cover >= len(nums) - 1 {
            return true
        }
    }
    return false
}
```

## [Maximum Bags With Full Capacity of Rocks](https://leetcode.com/problems/maximum-bags-with-full-capacity-of-rocks)

A: 每次选择余量最少的袋子装石头。

```cpp
class Solution {
public:
    int maximumBags(vector<int>& capacity, vector<int>& rocks, int additionalRocks) {
        int ans = 0;
        vector<int> rest;
        int len = capacity.size();
        for (int i = 0; i < len; i++) {
            rest.push_back(capacity[i] - rocks[i]);
        }
        sort(rest.begin(), rest.end());
        for (int i = 0; i < len; i++) {
            if (additionalRocks >= rest[i]) {
                additionalRocks -= rest[i];
                ans++;
            } else {
                break;
            }
        }
        return ans;
    }
};
```

## [Remove Stones to Minimize the Total](https://leetcode.com/problems/remove-stones-to-minimize-the-total)

A: 每次选择最大元素进行处理，用堆排序加快效率。

```cpp
int minStoneSum(vector<int>& A, int k) {
    priority_queue<int> pq(A.begin(), A.end());
    int res = accumulate(A.begin(), A.end(), 0);
    while (k--) {
        int a = pq.top();
        pq.pop();
        pq.push(a - a / 2);
        res -= a / 2;
    }
    return res;
}
```

## [Maximum Ice Cream Bars](https://leetcode.com/problems/maximum-ice-cream-bars)

A: 每次购买最便宜的冰激凌。

```cpp
int maxIceCream(vector<int>& costs, int coins) {
    sort(begin(costs), end(costs));
    for (int i = 0; i < costs.size(); ++i)
        if (coins >= costs[i])
            coins -= costs[i];
        else
            return i;
    return costs.size();
}
```

## [Jump Game II](https://leetcode.com/problems/jump-game-ii)

A: 最短路径问题，BFS，选择两次jump之后所能达到最远处的位置作为下一个起跳点。

```cpp
class Solution {
public:
    int jump(vector<int>& nums) {
        int n = nums.size();
        int result = 0;
        
        int i = 0;
        while (i < n - 1) {
            if (i + nums[i] >= n - 1) {
                result++;
                break;
            }
            int maxIndex = i + 1;
            int maxValue = 0;
            for (int j = i + 1; j < i + 1 + nums[i]; j++) {
                // j + nums[j]: 两次jump后的最大值
                if (j + nums[j] > maxValue) {
                    maxIndex = j;
                    maxValue = j + nums[j];
                }
            }
            i = maxIndex;
            result++;
        }
        
        return result;
    }
};
```

```go
func jump(nums []int) int {
    ans := 0
    q := []int {0} // stores index
    if len(nums) == 1 {
        return 0
    }
    for ; len(q) > 0; {
        ans++ // a new step
        for cnt := len(q); cnt > 0; cnt-- {
            cur := q[0] // current position
            q = q[1:]
            for i := nums[cur]; i > 0; i-- {
                // start from the farthest distance to trim
                next := cur + i
                if next < len(nums) - 1 {
                    if nums[next] != -1 {
                        q = append(q, next)
                    }
                } else {
                    return ans
                }
            }
            nums[cur] = -1 // label as visited
        }
    }
    return -1
}
```

A: Greedy，记录当前位置能到达的最远位置，当到达当前位置时，更新最远位置，同时步数+1。

```go
func jump(nums []int) int {
    ans, n, curEnd, curFar := 0, len(nums), 0, 0
    // initially, curEnd = 0, since we reach nums[0]
    for i := 0; i < n-1; i++ {
        // we have to update curFar timely to find
        // the farthest next position
        curFar = max(curFar, i + nums[i])
        if i == curEnd {
            // when we reach cuEnd, we've already
            // enumerated every possibility from the current position
            ans++
            if curFar >= n {
                // Once we can reach nums[n - 1], return!
                return ans
            }
            curEnd = curFar
        }
    }
    return ans
}

func max(x, y int) int {
    if x > y {
        return x
    }
    return y
}
```

## [Gas Station](https://leetcode.com/problems/gas-station)

A: 若A无法到达B，则A与B中所有中间点都无法达到B（A可以到达中间点）。

```cpp
class Solution {
public:
    int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
        int n = gas.size();
        
        int totalGas = 0;
        int totalCost = 0;
        for (int i = 0; i < n; i++) {
            totalGas += gas[i];
            totalCost += cost[i];
        }
        if (totalGas < totalCost) {
            return -1;
        }
        
        int total = 0;
        int result = 0;
        
        for (int i = 0; i < n; i++) {
            total += gas[i] - cost[i];
            if (total < 0) {
                total = 0;
                result = i + 1;
            }
        }
        
        return result;
    }
};
```

```go
func canCompleteCircuit(gas []int, cost []int) int {
	curSum := 0
	totalSum := 0
	start := 0
	for i := 0; i < len(gas); i++ {
		curSum += gas[i] - cost[i]
		totalSum += gas[i] - cost[i]
		if curSum < 0 {
			start = i+1
			curSum = 0
		}
	}
	if totalSum < 0 {
		return -1
	}
	return start
}
```

## [Hand of Straights](https://leetcode.com/problems/hand-of-straights)

A: 每次选择最小值作为起始值找卡片。

`map`默认以key升序排列。

```cpp
class Solution {
public:
    bool isNStraightHand(vector<int>& hand, int groupSize) {
        int n = hand.size();
        
        if (n % groupSize != 0) {
            return false;
        }
        
        // map {card value -> count}
        map<int, int> m;
        for (int i = 0; i < n; i++) {
            m[hand[i]]++;
        }
        
        while (!m.empty()) {
            int curr = m.begin()->first; // 最小值
            for (int i = 0; i < groupSize; i++) {
                if (m[curr + i] == 0) {
                    return false;
                }
                m[curr + i]--;
                if (m[curr + i] < 1) {
                    m.erase(curr + i);
                }
            }
        }
        
        return true;
    }
};
```

A: Go的map不保证遍历顺序，因此用最小堆来寻找最小数。

```go
type H []int

func (h H) Len() int { return len(h) }
func (h H) Less(i, j int) bool { return h[i] < h[j] }
func (h H) Swap(i, j int) { h[i], h[j] = h[j], h[i] }

func (h *H) Push(x interface{}) {
    *h = append(*h, x.(int))
}
func (h *H) Pop() interface{} {
    old := *h
    n := len(old)
    x := old[n-1]
    *h = old[:n-1]
    return x
}

func isNStraightHand(hand []int, groupSize int) bool {
    if len(hand) % groupSize != 0 {
        return false
    }
    m := map[int]int{}
    h := &H{}
    heap.Init(h)
    for i := range hand {
        heap.Push(h, hand[i])
        m[hand[i]]++
    }
    for h.Len() > 0 {
        cur := (*h)[0]
        if m[cur] > 0 {
            for i := 0; i < groupSize; i++ {
                if m[cur+i] == 0 {
                    return false
                } else {
                    m[cur+i]--
                }
            }
        }
        if m[cur] == 0 {
            heap.Pop(h)
        }
    }
    return true
}
```

## [Merge Triplets to Form Target Triplet](https://leetcode.com/problems/merge-triplets-to-form-target-triplet)

A: 跳过值大于target的三元组，在剩下三元组中选择满足单一条件的元组即可。

```cpp
class Solution {
public:
    bool mergeTriplets(vector<vector<int>>& triplets, vector<int>& target) {
        unordered_set<int> s;
        
        for (int i = 0; i < triplets.size(); i++) {
            if (triplets[i][0] > target[0] || triplets[i][1] > target[1] || triplets[i][2] > target[2]) {
                continue;
            }
            
            for (int j = 0; j < 3; j++) {
                if (triplets[i][j] == target[j]) {
                    s.insert(j);
                }
            }
        }
        
        return s.size() == 3;
    }
};
```

## [Partition Labels](https://leetcode.com/problems/partition-labels)

A: 记录每个字母最后出现的位置，遍历字符串找到最长子串。

```cpp
class Solution {
public:
    vector<int> partitionLabels(string s) {
        int n = s.size();
        // {char -> last index in s}
        vector<int> lastIndex(26);
        for (int i = 0; i < n; i++) {
            lastIndex[s[i] - 'a'] = i;
        }
        
        int size = 0;
        int end = 0;
        
        vector<int> result;
        
        for (int i = 0; i < n; i++) {
            size++;
            // constantly checking for further indices if possible
            end = max(end, lastIndex[s[i] - 'a']);
            if (i == end) {
                result.push_back(size);
                size = 0;
            }
        }
        
        return result;
    }
};
```

```go
func partitionLabels(s string) []int {
    m := map[rune]int{}
    for i, ch := range s {
        m[ch] = i
    }
    start, curMax := -1, 0
    ans := []int{}
    for i := 0; i < len(s); i++ {
        if i > curMax {
            ans = append(ans, curMax - start)
            start = curMax
        }
        if m[rune(s[i])] > curMax {
            curMax = m[rune(s[i])]
        }
    }
    ans = append(ans, curMax - start)
    return ans
}
```

## [Valid Parenthesis String](https://leetcode.com/problems/valid-parenthesis-string)

A: 将*看作(，得到所需)的最大数目，保持该数始终为正。

```cpp
class Solution {
public:
    bool checkValidString(string s) {
        int cmin = 0, cmax = 0;
        for (char c : s) {
            if (c == '(')
                cmax++, cmin++;
            else if (c == ')')
                cmax--, cmin = max(cmin - 1, 0);
            else
                cmax++, cmin = max(cmin - 1, 0);
            if (cmax < 0) return false;
        }
        return cmin == 0;
    }
};
```

## [Jump Game VII](https://leetcode.com/problems/jump-game-vii)

A: BFS。

```cpp
class Solution {
public:
    bool canReach(string s, int minJump, int maxJump) {
        int n = s.length();
        if(s[n-1]!='0')
            return false;
        
        int i = 0;
        queue<int> q;
        q.push(0);
        int curr_max = 0;
        
        while(!q.empty()){
            i = q.front();
            q.pop();
            if(i == n-1)
                return true;
            
            for(int j = max(i + minJump, curr_max); j <= min(i + maxJump, n - 1); j++){
                if(s[j] == '0')   q.push(j);
            }   
            curr_max = min(i+maxJump+1, n);
        }
        return false;
    }
};
```

## [Eliminate Maximum Number of Monsters](https://leetcode.com/problems/eliminate-maximum-number-of-monsters)

A: 按照到来顺序排序，先击杀最早到的。

```cpp
class Solution {
public:
    int eliminateMaximum(vector<int>& dist, vector<int>& speed) {
        for (int i = 0; i < dist.size(); ++i)
            dist[i] = (dist[i] - 1) / speed[i];
        sort(begin(dist), end(dist));
        for (int i = 0; i < dist.size(); ++i)
            if (i > dist[i])
                return i;
        return dist.size();
    } 
};
```

## [Two City Scheduling](https://leetcode.com/problems/two-city-scheduling)

A: 维护两个优先队列，排序逻辑为调整目的地付出代价升序。

```cpp
struct cmp {
    bool operator() (vector<int> &lhs, vector<int> &rhs){
        return abs(lhs[1] - lhs[0]) > abs(rhs[1] - rhs[0]);
    }
};

class Solution {
public:
    int twoCitySchedCost(vector<vector<int>>& costs) {
        priority_queue<vector<int>, vector<vector<int>>, cmp> pqA;
        priority_queue<vector<int>, vector<vector<int>>, cmp> pqB;
        int sumA = 0, sumB = 0;
        for (auto p : costs) {
            if (p[0] <= p[1]) {
                pqA.push(p);
                sumA += p[0];
            } else {
                pqB.push(p);
                sumB += p[1];
            }
        }
        int n = costs.size() / 2;
        while (pqA.size() > n) {
            vector<int> cost = pqA.top();
            pqA.pop();
            sumA -= cost[0];
            sumB += cost[1];
        }
        while (pqB.size() > n) {
            vector<int> cost = pqB.top();
            pqB.pop();
            sumA += cost[0];
            sumB -= cost[1];
        }
        return sumA + sumB;
    }
};
```

## [Assign Cookies](https://leetcode.com/problems/assign-cookies)

A: 优先满足食量小的孩子。

```go
func findContentChildren(g []int, s []int) int {
    sort.Ints(g)
    sort.Ints(s)
    ans := 0
    i, j := 0, 0
    for i < len(s) && j < len(g) {
        if s[i] >= g[j] {
            ans++
            i++
            j++
        } else  {
            i++
        }
    }
    return ans
}
```

## [Wiggle Subsequence](https://leetcode.com/problems/wiggle-subsequence)

A: 忽略所有单调坡和平坡，只计算峰和谷。

```go
func wiggleMaxLength(nums []int) int {
	n := len(nums)
	if n < 2 {
		return n
	}
	ans := 1
	prevDiff := nums[1] - nums[0]
	if prevDiff != 0 {
		ans = 2
	}
	for i := 2; i < n; i++ {
		diff := nums[i] - nums[i-1]
		if diff > 0 && prevDiff <= 0 || diff < 0 && prevDiff >= 0 {
			ans++
			prevDiff = diff
		}
	}
	return ans
}
```

## [Maximize Sum Of Array After K Negations](https://leetcode.com/problems/maximize-sum-of-array-after-k-negations)

A: 每次取最小的数字取反。

```go
func largestSumAfterKNegations(nums []int, K int) int {
	sort.Slice(nums, func(i, j int) bool {
		return math.Abs(float64(nums[i])) > math.Abs(float64(nums[j]))
	})
  
	for i := 0; i < len(nums); i++ {
		if K > 0 && nums[i] < 0 {
			nums[i] = -nums[i]
			K--
		}
	}

	if K%2 == 1 {
		nums[len(nums)-1] = -nums[len(nums)-1]
	}

	result := 0
	for i := 0; i < len(nums); i++ {
		result += nums[i]
	}
	return result
}
```

A: 按绝对值逆序排列，优先变负为正，其次将绝对值最小的正数取反。

```go
func largestSumAfterKNegations(nums []int, K int) int {
	sort.Slice(nums, func(i, j int) bool {
		return math.Abs(float64(nums[i])) > math.Abs(float64(nums[j]))
	})
  
	for i := 0; i < len(nums); i++ {
		if K > 0 && nums[i] < 0 {
			nums[i] = -nums[i]
			K--
		}
	}

	if K%2 == 1 {
		nums[len(nums)-1] = -nums[len(nums)-1]
	}

	result := 0
	for i := 0; i < len(nums); i++ {
		result += nums[i]
	}
	return result
}
```

## [Candy](https://leetcode.com/problems/candy)

A: 两遍遍历，首先从左往右遍历，遇到递增就多分配一个糖果。然后相同策略反向遍历，取两次遍历的最大值。

```go
func candy(ratings []int) int {
    /**先确定一边，再确定另外一边
        1.先从左到右，当右边的大于左边的就加1
        2.再从右到左，当左边的大于右边的就再加1
    **/
    need := make([]int, len(ratings))
    sum := 0
    // 初始化(每个人至少一个糖果)
     for i := 0; i < len(ratings); i++ {
         need[i] = 1
     }
     // 1.先从左到右，当右边的大于左边的就加1
    for i := 0; i < len(ratings) - 1; i++ {
        if ratings[i] < ratings[i+1] {
            need[i+1] = need[i] + 1
        }
    }
    // 2.再从右到左，当左边的大于右边的就右边加1，但要花费糖果最少，所以需要做下判断
    for i := len(ratings)-1; i > 0; i-- {
        if ratings[i-1] > ratings[i] {
            need[i-1] = findMax(need[i-1], need[i]+1)
        }
    }
    //计算总共糖果
    for i := 0; i < len(ratings); i++ {
        sum += need[i]
    }
    return sum
}
func findMax(num1 int, num2 int) int {
    if num1 > num2 {
        return num1
    }
    return num2
}
```

## [Lemonade Change](https://leetcode.com/problems/lemonade-change)

A: 优先消耗面额大的零钱，因为面额小的更万能，可以组成大的数额。

```go
func lemonadeChange(bills []int) bool {
    m := map[int]int{}

    for _, v := range bills {
        m[v]++
        switch v {
            case 10:
                m[5]--
                if m[5] < 0 {
                    return false
                }
            case 20:
                if m[10] > 0 && m[5] > 0 {
                    m[10]--
                    m[5]--
                } else if m[5] > 0 {
                    m[5] -= 3
                } else {
                    return false
                }
                if m[10] < 0 || m[5] < 0 {
                    return false
                }
        }
    }
    return true
}
```

## [Queue Reconstruction by Height](https://leetcode.com/problems/queue-reconstruction-by-height)

A: 首先按照身高从大到小排序，然后按照K进行插入排序，优先插入K小的。

```go
func reconstructQueue(people [][]int) [][]int {
    // 先将身高从大到小排序，确定最大个子的相对位置
    sort.Slice(people, func(i, j int) bool {
        if people[i][0] == people[j][0] {
            return people[i][1] < people[j][1]   // 当身高相同时，将K按照从小到大排序
        }
        return people[i][0] > people[j][0]     // 身高按照由大到小的顺序来排
    })

    // 再按照K进行插入排序，优先插入K小的
	for i, p := range people {
		copy(people[p[1]+1 : i+1], people[p[1] : i+1])  // 空出一个位置
		people[p[1]] = p
	}
	return people
}
```

## [Monotone Increasing Digits](https://leetcode.com/problems/monotone-increasing-digits)

A: 反向遍历，当前一个大于后一位时，前一位减1，后面的全部置为9。

```go
func monotoneIncreasingDigits(N int) int {
    s := strconv.Itoa(N)//将数字转为字符串，方便使用下标
    ss := []byte(s)//将字符串转为byte数组，方便更改。
    n := len(ss)
    if n <= 1 {
        return N
    }
    for i := n-1; i > 0; i-- {
        if ss[i-1] > ss[i] {   //前一个大于后一位,前一位减1，后面的全部置为9
            ss[i-1] -= 1
            for j := i; j < n; j++ {   //后面的全部置为9
                ss[j] = '9'
            }
        } 
    }
    res, _ := strconv.Atoi(string(ss))
    return res 
}
```

## [Split With Minimum Sum](https://leetcode.com/problems/split-with-minimum-sum)

A: 将数字分解，然后排序，从小到大依次相加。

```go
func splitNum(num int) int {
    num1, num2 := 0, 0
    nums := []int{}
    for num != 0 {
        nums = append(nums, num % 10)
        num /= 10
    }
    sort.Ints(nums)
    for i := 0; i < len(nums); i++ {
        num1 = num1 * 10 + nums[i]
        num1, num2 = num2, num1
    }
    return num1 + num2
}
```

## [Rearrange Array to Maximize Prefix Score](https://leetcode.com/problems/rearrange-array-to-maximize-prefix-score)

A: 降序排列。

```go
func maxScore(nums []int) int {
    sort.Slice(nums, func(i, j int) bool {
        return nums[i] > nums[j]
    })
    prefix := 0
    ans := 0
    for _, v := range nums {
        prefix += v
        if prefix > 0 {
            ans++
        }
    }
    return ans
}
```

## [Reducing Dishes](https://leetcode.com/problems/reducing-dishes)

A: 降序排列，然后从前往后累加。通过suffixSum的累加实现了较大数字的多次累加。

```go
func maxSatisfaction(satisfaction []int) int {
    sort.Ints(satisfaction)
    maxSatisfaction, suffixSum := 0, 0
    for i := len(satisfaction)-1; i >= 0 && suffixSum + satisfaction[i] > 0; i-- {
        suffixSum += satisfaction[i]
        maxSatisfaction += suffixSum
    }
    return maxSatisfaction
}
```

## [Optimal Partition of String](https://leetcode.com/problems/optimal-partition-of-string)

A: 用map记录每个字符出现的次数，当遇到重复的字符时，将map清空，计数器加1。

```go
func partitionString(s string) int {
    set := map[rune]struct{}{}
    ans := 0
    for _, ch := range s {
        if _, ok := set[ch]; ok {
            ans++
            set = map[rune]struct{}{}
        }
        set[ch] = struct{}{}
    }
    if len(set) > 0 {
        ans++
    }
    return ans
}
```

## [Split a String in Balanced Strings](https://leetcode.com/problems/split-a-string-in-balanced-strings)

A: 贪心，满足条件则立即切分。

```go
func balancedStringSplit(s string) int {
    diff := 0 // 右左差值
    ans := 0
    for _, c := range s {
        if c == 'L' {
            diff--
        }else {
            diff++
        }
        if diff == 0 {
            ans++
        }
    }
    return ans
}
```

## [Kids With the Greatest Number of Candies](https://leetcode.com/problems/kids-with-the-greatest-number-of-candies)

A: 找出最大值，遍历数组，判断是否满足条件。

```cpp
class Solution {
public:
    vector<bool> kidsWithCandies(vector<int>& candies, int extraCandies) {
        vector<bool> ans(candies.size(), false);
        int m = *max_element(candies.begin(), candies.end());
        m -= extraCandies;
        for (int i = 0; i < candies.size(); i++) {
            if (candies[i] >= m) {
                ans[i] = true;
            }
        }
        return ans;
    }
};
```

```go
func kidsWithCandies(candies []int, extraCandies int) []bool {
    ans := make([]bool, len(candies))
    min := -1
    for _, c := range candies {
        n := c - extraCandies
        if n > min {
            min = n
        }
    }
    for i := range candies {
        if candies[i] >= min {
            ans[i] = true
        }
    }
    return ans
}
```
