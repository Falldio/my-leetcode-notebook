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
