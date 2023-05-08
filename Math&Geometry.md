# Math & Geometry

## [Rotate Image](https://leetcode.com/problems/rotate-image)

A: DFS。

```cpp
class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int len = matrix.size();
        vector<vector<bool>> rotated(len, vector<bool>(len, false));

        for (int i = 0; i < len; i++) {
            for (int j = 0; j < len; j++) {
                dfs(i, j, matrix, rotated, matrix[i][j]);
            }
        }
    }

private:
    void dfs(int i, int j, vector<vector<int>>& matrix, vector<vector<bool>>& rotated, int value) {
        int len = matrix.size();
        if (rotated[j][len - 1- i])  return;

        int tmp = matrix[j][len - 1 - i];
        matrix[j][len - 1 - i] = value;
        rotated[j][len - 1 - i] = true;
        dfs(j, len - 1 - i, matrix, rotated, tmp);
    }
};
```

A: Transpose + reflect (rev on diag then rev left to right)。

```cpp
class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int n = matrix.size();
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                swap(matrix[i][j], matrix[j][i]);
            }
            reverse(matrix[i].begin(), matrix[i].end());
        }
    }
};
```

```go
func rotate(matrix [][]int)  {
    for i := 0; i < len(matrix); i++ {
        for j := i; j < len(matrix); j++ {
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        }
    }
    for i := 0; i < len(matrix); i++ {
        for j := 0; j < len(matrix) / 2; j++ {
            matrix[i][j], matrix[i][len(matrix) - 1 - j] = matrix[i][len(matrix) - 1 - j], matrix[i][j]
        }
    }
}
```

## [Spiral Matrix](https://leetcode.com/problems/spiral-matrix)

A: 用边界限制遍历条件和循环条件。

```cpp
class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        int left = 0;
        int top = 0;
        int right = matrix[0].size() - 1;
        int bottom = matrix.size() - 1;
        
        vector<int> result;
        
        while (top <= bottom && left <= right) {
            for (int j = left; j <= right; j++) {
                result.push_back(matrix[top][j]);
            }
            top++;
            
            for (int i = top; i <= bottom; i++) {
                result.push_back(matrix[i][right]);
            }
            right--;
            
            if (top <= bottom) {
                for (int j = right; j >= left; j--) {
                    result.push_back(matrix[bottom][j]);
                }
            }
            bottom--;
            
            if (left <= right) {
                for (int i = bottom; i >= top; i--) {
                    result.push_back(matrix[i][left]);
                }
            }
            left++;
        }
        
        return result;
    }
};
```

```go
func spiralOrder(matrix [][]int) []int {
    m, n := len(matrix), len(matrix[0])
    l, r, t, b := 0, n - 1, 0, m - 1
    ans := make([]int, 0, m * n)
    for len(ans) < m * n {
        if t <= b {
            for j := l; j <= r; j++ {
                ans = append(ans, matrix[t][j])
            }
            t++
        }
        if l <= r {
            for i := t; i <= b; i++ {
                ans = append(ans, matrix[i][r])
            }
            r--
        }
        if t <= b {
            for j := r; j >= l; j-- {
                ans = append(ans, matrix[b][j])
            }
            b--
        }
        if l <= r {
            for i := b; i >= t; i-- {
                ans = append(ans, matrix[i][l])
            }
            l++
        }
    }
    return ans
}
```

## [Set Matrix Zeroes](https://leetcode.com/problems/set-matrix-zeroes)

A: 用第一行+第一列来标记对应的行列是否有0，最小的空间复杂度。

```cpp
class Solution {
public:
    void setZeroes(vector<vector<int>>& matrix) {
        int m = matrix.size();
        int n = matrix[0].size();
        
        bool isFirstRowZero = false;
        bool isFirstColZero = false;
        
        for (int i = 0; i < m; i++) {
            if (matrix[i][0] == 0) {
                isFirstColZero = true;
                break;
            }
        }
        
        for (int j = 0; j < n; j++) {
            if (matrix[0][j] == 0) {
                isFirstRowZero = true;
                break;
            }
        }
        
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (matrix[i][j] == 0) {
                    matrix[i][0] = 0;
                    matrix[0][j] = 0;
                }
            }
        }
        
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (matrix[i][0] == 0 || matrix[0][j] == 0) {
                    matrix[i][j] = 0;
                }
            }
        }
        
        if (isFirstColZero) {
            for (int i = 0; i < m; i++) {
                matrix[i][0] = 0;
            }
        }
        
        if (isFirstRowZero) {
            for (int j = 0; j < n; j++) {
                matrix[0][j] = 0;
            }
        }
    }
};
```

## [Minimum Rounds to Complete All Tasks](https://leetcode.com/problems/minimum-rounds-to-complete-all-tasks)

A: 完成任务轮数和任务数目之前存在函数关系，直接计算。

```cpp
class Solution {
public:
    int minimumRounds(vector<int>& tasks) {
        int ans = 0;
        unordered_map<int, int> m;
        for (auto t : tasks) {
            m[t]++;
        }
        for (auto i : m) {
            int freq = i.second;
            if (freq == 1) return -1;
            if (freq % 3 == 0) ans += freq / 3;
            else {
                ans += freq / 3 + 1;
            }
        }
        return ans;
    }
};
```

## [Max Points on a Line](https://leetcode.com/problems/max-points-on-a-line)

A: 用斜率（字符串存储斜率，避免pair<double,double>无法哈希）和线上一点确定一条线

```cpp
class Solution {
public:
    int maxPoints(vector<vector<int>>& points) {
        int n = points.size(), ans = 0;
        for (int i = 0; i < n; i++) {
            unordered_map<string, int> counter;
            // 最外层循环确定线的标定点
            int dup = 1; // 初始有一个点
            for (int j = i + 1; j < n; j++) {
                int dx = points[j][0] - points[i][0], dy = points[j][1] - points[i][1], g = gcd(dx, dy);
                counter[to_string(dx / g) + '_' + to_string(dy / g)]++;
            }
            for (auto p : counter) {
                ans = max(ans, p.second + dup);
            }
        }
        return ans;
    }
private:
    int gcd(int a, int b) {
        while (b) {
            a = a % b;
            swap(a, b);
        }
        return a;
    }
};
```

## [Happy Number](https://leetcode.com/problems/happy-number)

A: 若非快乐数，则运算过程中会出现重复数字，问题为快慢指针检测循环。

```cpp
class Solution {
public:
    bool isHappy(int n) {
        int slow = n;
        int fast = getNext(n);
        
        while (slow != fast && fast != 1) {
            slow = getNext(slow);
            fast = getNext(getNext(fast));
        }
        
        if (fast == 1) {
            return true;
        }
        return false;
    }
private:
    int getNext(int n) {
        int sum = 0;
        while (n > 0) {
            int digit = n % 10;
            n /= 10;
            sum += pow(digit, 2);
        }
        return sum;
    }
};
```

## [Plus One](https://leetcode.com/problems/plus-one)

A: 从右往左依次加1。

```cpp
class Solution {
public:
    vector<int> plusOne(vector<int>& digits) {
        for (int i = digits.size() - 1; i >= 0; i--) {
            if (digits[i] < 9) {
                digits[i]++;
                return digits;
            }
            digits[i] = 0;
        }
        // 此时意味着digits全部为9，需要增加末尾0，改最左边数字为1
        digits[0] = 1;
        digits.push_back(0);
        return digits;
    }
};
```

## [Pow(x, n)](https://leetcode.com/problems/powx-n)

A: 用平方加快运算速度（i/=2）。

```cpp
class Solution {
public:
    double myPow(double x, int n) {
        long exponent = abs(n);
        double curr = x;
        double result = 1.0;
        
        for (long i = exponent; i > 0; i /= 2) {
            if (i % 2 == 1) {
                result *= curr;
            }
            curr *= curr;
        }
        
        if (n < 0) {
            return 1.0 / result;
        }
        return result;
    }
};
```

## [Multiply Strings](https://leetcode.com/problems/multiply-strings)

A: 竖式乘法。

```cpp
class Solution {
public:
    string multiply(string num1, string num2) {
        int m = num1.size();
        int n = num2.size();
        
        string result(m + n, '0');
        
        for (int i = m - 1; i >= 0; i--) {
            for (int j = n - 1; j >= 0; j--) {
                int sum = (num1[i] - '0') * (num2[j] - '0') + (result[i + j + 1] - '0');
                result[i + j + 1] = sum % 10 + '0';
                result[i + j] += sum / 10;
            }
        }
        
        for (int i = 0; i < m + n; i++) {
            if (result[i] != '0') {
                return result.substr(i);
            }
        }
        return "0";
    }
};
```

## [Detect Squares](https://leetcode.com/problems/detect-squares)

A: 首先寻找对角点，根据对角点获取其余点的个数。

```cpp
class DetectSquares {
public:
    DetectSquares() {
        
    }
    
    void add(vector<int> point) {
        points[point[0]][point[1]]++;
    }
    
    int count(vector<int> point) {
        int x1 = point[0];
        int y1 = point[1];
        
        int result = 0;
        
        for (auto x = points.begin(); x != points.end(); x++) {
            unordered_map<int, int> yPoints = x->second;
            for (auto y = yPoints.begin(); y != yPoints.end(); y++) {
                int x3 = x->first;
                int y3 = y->first;
                
                // skip points on same x-axis or y-axis
                if (abs(x3 - x1) == 0 || abs(x3 - x1) != abs(y3 - y1)) {
                    continue;
                }
                
                result += points[x3][y3] * points[x1][y3] * points[x3][y1];
            }
        }
        
        return result;
    }
private:
    // {x -> {y -> count}}
    unordered_map<int, unordered_map<int, int>> points;
};
```

## [Palindrome Number](https://leetcode.com/problems/palindrome-number)

A: 将x一分为二，一半逆序，对比两个数是否相等。

```cpp
class Solution {
public:
    bool isPalindrome(int x) {
        if (x < 0 || (x % 10 == 0 && x != 0)) return false;
        int rev = 0;
        while (rev < x) {
            rev = rev * 10 + x % 10;
            x /= 10;
        }
        
        return x == rev || x == rev / 10;
    }
};
```

## [Robot Bounded In Circle](https://leetcode.com/problems/robot-bounded-in-circle)

A: 当且仅当回到原点或者方向改变时，轨迹能被圈住。

```cpp
class Solution {
public:
    bool isRobotBounded(string instructions) {
        int x = 0, y = 0, i = 0;
        vector<vector<int>> d = {{0, 1}, {1, 0}, {0, -1}, { -1, 0}};
        for (char & ins : instructions)
            if (ins == 'R')
                i = (i + 1) % 4;
            else if (ins == 'L')
                i = (i + 3) % 4;
            else
                x += d[i][0], y += d[i][1];
        return x == 0 && y == 0 || i > 0;
    }
};
```

## [Integer to Roman](https://leetcode.com/problems/integer-to-roman)

A: 按照位数赋予罗马数字。

```cpp
class Solution {
public:
    string intToRoman(int num) {
        string ones[] = {"","I","II","III","IV","V","VI","VII","VIII","IX"};
        string tens[] = {"","X","XX","XXX","XL","L","LX","LXX","LXXX","XC"};
        string hrns[] = {"","C","CC","CCC","CD","D","DC","DCC","DCCC","CM"};
        string ths[]={"","M","MM","MMM"};
        
        return ths[num/1000] + hrns[(num%1000)/100] + tens[(num%100)/10] + ones[num%10];
    }
};
```

## [Zigzag Conversion](https://leetcode.com/problems/zigzag-conversion)

A: 用step控制行方向。

```cpp
class Solution {
public:
    string convert(string s, int numRows) {
        if(numRows <= 1) return s;
        vector<string> db (numRows, "");
        for(int i = 0,row = 0,step = 1;i < s.size();i++){
            db[row] += s[i];
            if(row == 0) step = 1;
            if(row == numRows - 1) step = -1;
            row += step;
        }
        string ret;
        for(auto d:db) ret+=d;
        return ret;
    }
};
```

## [Find Missing Observations](https://leetcode.com/problems/find-missing-observations)

A: 判断剩余骰子数值总数是否符合实际，生成可能的结果。

```cpp
class Solution {
public:
    vector<int> missingRolls(vector<int>& rolls, int mean, int n) {
        int sumM = accumulate(rolls.begin(), rolls.end(), 0);
        int total = mean * (rolls.size() + n);
        int sumN = total - sumM;
        if (sumN > 6 * n || sumN < n) return vector<int> ();
        int cur = 0, ave = sumN / n;
        vector<int> ans(n, ave);
        int rest = sumN % n;
        int i = 0;
        while (rest && i < n) {
            int old = ans[i];
            ans[i] = min(6, rest + ans[i]);
            rest -= ans[i] - old;
            i++;
        }
        return ans;
    }
};
```

## [Ugly Number](https://leetcode.com/problems/ugly-number)

A: 除以条件中尽可能大的质因数。

```cpp
class Solution {
public:
    bool isUgly(int n) {
        if (n <= 0) return false;
        for (int i=2; i<6 && n; i++)
            while (n % i == 0)
                n /= i;
        return n == 1;
    }
};
```

## [Shift 2D Grid](https://leetcode.com/problems/shift-2d-grid)

A: 重新构造矩阵，计算拷贝后的元素位置。

```cpp
class Solution {
public:
    vector<vector<int>> shiftGrid(vector<vector<int>>& grid, int k) {
        int n=grid.size();
        int m=grid[0].size();
        vector<vector<int>> ans(n,vector<int>(m));
        
        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                
                int newJ=(j+k)%m; // (j + numbers of columns added)%m
                
                int newI=(i+(j+k)/m)%n; // (i + numbers of rows added)%n 
                
                ans[newI][newJ]=grid[i][j];
            }
        }
        return ans;
    }
};
```

## [Count Total Number of Colored Cells](https://leetcode.com/problems/count-total-number-of-colored-cells)

A: 想象成坐标系，先计算x，y，在计算四个象限的网格数目。

```go
func coloredCells(n int) int64 {
    var ans int64
    // First we compute two cross lines in the middle
    ans += (2 * (int64(n) - 1) + 1) * 2 - 1
    if n >= 3 {
    // Then we add the nums in the 4 quadrants
        ans += 2 * (int64(n) - 1) * (int64(n) - 2)
    }
    return ans
}
```

## [Linked List Random Node](https://leetcode.com/problems/linked-list-random-node)

A: 暴力得到长度，然后取随机idx。

```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
type Solution struct {
    head *ListNode
    cnt int
}


func Constructor(head *ListNode) Solution {
    cur := head
    cnt := 0
    for cur != nil {
        cnt++
        cur = cur.Next
    }
    return Solution{head, cnt}
}


func (this *Solution) GetRandom() int {
    idx := rand.Intn(this.cnt)
    cur := this.head
    for idx != 0 {
        cur = cur.Next
        idx--
    }
    return cur.Val
}


/**
 * Your Solution object will be instantiated and called as such:
 * obj := Constructor(head);
 * param_1 := obj.GetRandom();
 */
 ```

 A: Reservoir Sampling。

[详解](https://leetcode.com/problems/linked-list-random-node/solutions/85659/brief-explanation-for-reservoir-sampling/)

```go
type Solution struct {
	head *ListNode
}

func Constructor(head *ListNode) Solution {
	return Solution{head: head}
}

func (this *Solution) GetRandom() int {
	cnt, node, candidate := 0, this.head, this.head

	for node != nil {
		cnt++
        // 1/k的概率替换
		if rand.Intn(cnt) == 0 {
			candidate = node
		}
		node = node.Next
	}

	return candidate.Val
}
```

## [Dota2 Senate](https://leetcode.com/problems/dota2-senate)

A: 模拟过程。

```go
func predictPartyVictory(senateStr string) string {
	// R = true表示本轮循环结束后，字符串里依然有R。D同理
	R, D := true, true
	// 当flag大于0时，R在D前出现，R可以消灭D。当flag小于0时，D在R前出现，D可以消灭R
	flag := 0

	senate := []byte(senateStr)
	for R && D { // 一旦R或者D为false，就结束循环，说明本轮结束后只剩下R或者D了
		R = false
		D = false
		for i := 0; i < len(senate); i++ {
			if senate[i] == 'R' {
				if flag < 0  {
					 senate[i] = 0  // 消灭R，R此时为false
				} else {
					R = true // 如果没被消灭，本轮循环结束有R
				}
				flag++;
			}
			if (senate[i] == 'D') {
				if flag > 0 {
					senate[i] = 0
				} else  {
					D = true
				}
				flag--
			}
		}
	}
	// 循环结束之后，R和D只能有一个为true
	if R {
		return "Radiant"
	}
	return "Dire";
}
```

```cpp
class Solution {
public:
    string predictPartyVictory(string senate) {
        bool R = true, D = true;
        int flag = 0; // -1: R D, 1: D, R
        while (R && D) {
            R = false;
            D = false;
            for (int i = 0; i < senate.size(); i++) {
                if (senate[i] == 'R') {
                    if (flag > 0) {
                        senate[i] = '#';
                    } else {
                        R = true;
                    }
                    flag--;
                }
                if (senate[i] == 'D') {
                    if (flag < 0) {
                        senate[i] = '#';
                    } else {
                        D = true;
                    }
                    flag++;
                }
            }
        }
        if (R) {
            return "Radiant";
        } else {
            return "Dire";
        }
    }
};
```

# [Robot Return to Origin](https://leetcode.com/problems/robot-return-to-origin)

A: 记录两个方向的坐标偏移。

```go
func judgeCircle(moves string) bool {
    x, y := 0, 0
    for _, m := range moves {
        switch m {
            case 'U': y++
            case 'D': y--
            case 'L': x--
            case 'R': x++
        }
    }
    return x == 0 && y == 0
}
```

## [Minimum Number of Operations to Make All Array Elements Equal to 1](https://leetcode.com/problems/minimum-number-of-operations-to-make-all-array-elements-equal-to-1)

A: **1和其它元素的gcd为1**，因此，如果数组中有c个1，则答案为n-c。否则，枚举每个元素，计算其与其后面元素的gcd，如果gcd为1，则答案为j-i+(n-1)。

```cpp
class Solution {
public:
    int minOperations(vector<int>& nums) {
        int n=nums.size();
        int c=count(nums.begin(),nums.end(),1);     // if there is at least 1. then it's easy!
        if(c!=0)    return n-c;
        int res=1e7;        // just a dummy value
        for(int i=0;i<n;i++){
            int g=nums[i];
            for(int j=i+1;j<n;j++){
                g=__gcd(g,nums[j]);
                if(g==1){
                    res=min(res,j-i+(n-1)); // number of operations to make this element 1+ number of non ones (i.e. n-1) .
                    break;
                } 
            }
        }
        return res==1e7?-1:res;
    }
};
```

## [Satisfiability of Equality Equations](https://leetcode.com/problems/satisfiability-of-equality-equations)

A: 并查集，先处理等式，再处理不等式。

```go
var (
    parent []int
)

func equationsPossible(equations []string) bool {
    parent = make([]int, 26)
    for i := range parent {
        parent[i] = i
    }
    for _, e := range equations {
        if e[1] == '!' {
            continue
        }
        union(int(e[0] - 'a'), int(e[3] - 'a'))
    }
    fmt.Println(parent)
    for _, e := range equations {
        if e[1] == '=' {
            continue
        }
        if connected(int(e[0] - 'a'), int(e[3] - 'a')) {
            return false
        }
    }
    return true
}

func union(n1, n2 int) {
    p1, p2 := find(n1), find(n2)
    if p1 == p2 {
        return
    } else if p1 < p2 {
        parent[p1] = p2
    } else {
        parent[p2] = p1
    }
}

func find(x int) int {
    if x != parent[x] {
        parent[x] = find(parent[x])
    }
    return parent[x]
}

func connected(n1, n2 int) bool {
    return find(n1) == find(n2)
}
```
