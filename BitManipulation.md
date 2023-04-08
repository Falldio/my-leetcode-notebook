# Bit Manipulation

## [Number of 1 Bits](https://leetcode.com/problems/number-of-1-bits)

A: `n - 1`的操作实际上将`n`最右边的1消去了，因此`n &= (n - 1)`的运行次数等于`n`中1的个数。

```cpp
class Solution {
public:
    int hammingWeight(uint32_t n) {
        int count = 0;

        while (n) {
            n &= (n - 1);
            count++;
        }

        return count;
    }
};
```

## [Sort Integers by The Number of 1 Bits](https://leetcode.com/problems/sort-integers-by-the-number-of-1-bits)

A: 重写sort，按照1的个数排序，如果相同则按照数字大小排序。

```go
func sortByBits(arr []int) []int {
    sort.Slice(arr, func(i, j int) bool {
        n1, n2 := cnt(arr[i]), cnt(arr[j])
        if n1 != n2 {
            return n1 < n2
        } else {
            return arr[i] < arr[j]
        }
    })
    return arr
}

func cnt(i int) int {
    ans := 0

    for i != 0 {
        i &= (i - 1)
        ans++
    }

    return ans
}
```

## [Counting Bits](https://leetcode.com/problems/counting-bits)

A: `n - 1`的操作实际上将`n`最右边的1消去了，因此`n &= (n - 1)`的运行次数等于`n`中1的个数。

```cpp
class Solution {
public:
    vector<int> countBits(int n) {
        vector<int> ans(n + 1, 0);

        for (int i = 0; i < n + 1; i++) {
            int j = i;
            while (j) {
                j &= j - 1;
                ans[i]++;
            }
        }
        
        return ans;
    }
};
```

## [Reverse Bits](https://leetcode.com/problems/reverse-bits)

A: 32次循环，每次取`n`的最右位进行与运算得到bit，再通过或运算赋值给返回值。

```cpp
class Solution {
public:
    uint32_t reverseBits(uint32_t n) {
        uint32_t result = 0;
        
        for (int i = 0; i < 32; i++) {
            result <<= 1;
            result |= n & 1;
            n >>= 1;
        }
        
        return result;
    }
};
```

## [Missing Number](https://leetcode.com/problems/missing-number)

A: 全部异或(`{i}与{nums[i]}`)之后只有缺失数字保留。

```cpp
class Solution {
public:
    int missingNumber(vector<int>& nums) {
        int n = nums.size();
        int result = n;
        for (int i = 0; i < n; i++) {
            result ^= i ^ nums[i];
        }
        return result;
    }
};
```

## [Sum of Two Integers](https://leetcode.com/problems/sum-of-two-integers)

A: `a & b`检查进位，`(a & b) << 1`将进位进1，`a ^ b`检查保留位。

[详解](https://leetcode.com/problems/sum-of-two-integers/solutions/84278/a-summary-how-to-use-bit-manipulation-to-solve-problems-easily-and-efficiently/?orderBy=most_votes)

```cpp
class Solution {
public:
    int getSum(int a, int b) {
        while (b != 0) {
            int carry = a & b;
            a = a ^ b;
            b = (unsigned)carry << 1;
        }
        return a;
    }
};
```

## [Single Number](https://leetcode.com/problems/single-number)

A: 数组中的单独数字无法被异或归零。

```cpp
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int n = nums[0];
        for (int i = 1; i < nums.size(); i++) {
            n ^= nums[i];
        }
        return n;
    }
};
```

## [Reverse Integer](https://leetcode.com/problems/reverse-integer)

A: 检查最后一位和极限值最后一位的大小来决定是否超限。

```cpp
class Solution {
public:
    int reverse(int x) {
        int rev = 0;
        while (x != 0) {
            int temp = x % 10;
            x /= 10;
            if (rev > INT_MAX / 10 || (rev == INT_MAX / 10 && temp > 7)) {
                return 0;
            }
            if (rev < INT_MIN / 10 || (rev == INT_MIN / 10 && temp < -8)) {
                return 0;
            }
            rev = rev * 10 + temp;
        }
        return rev;
    }
};
```

## [Add Binary](https://leetcode.com/problems/add-binary)

A: 注意代码格式。

```cpp
class Solution {
public:
    string addBinary(string a, string b) {
        string s = "";
        
        int c = 0, i = a.size() - 1, j = b.size() - 1;
        while(i >= 0 || j >= 0 || c == 1) {
            c += i >= 0 ? a[i--] - '0' : 0;
            c += j >= 0 ? b[j--] - '0' : 0;
            s = char(c % 2 + '0') + s;
            c /= 2;
        }
        
        return s;
    }
};
```

```go
// 代码可以精简为一个for
func addBinary(a string, b string) string {
    i, j := len(a) - 1, len(b) - 1
    ans := ""
    pre := 0 // add digit
    for i >= 0 && j >= 0 {
        sum := int(a[i] - '0') + int(b[j] - '0') + pre
        ans = strconv.Itoa(sum % 2) + ans
        pre = sum / 2
        i--
        j--
    }
    for i >= 0 {
        sum := int(a[i] - '0') + pre
        ans = strconv.Itoa(sum % 2) + ans
        pre = sum / 2
        i--
    }
    for j >= 0 {
        sum := int(b[j] - '0') + pre
        ans = strconv.Itoa(sum % 2) + ans
        pre = sum / 2
        j--
    }
    if pre != 0 {
        ans = strconv.Itoa(pre) + ans
    }
    return ans
}
```

## [Count the Number of Beautiful Subarrays](https://leetcode.com/problems/count-the-number-of-beautiful-subarrays)

A: 题目条件实为异或操作，统计`xor = 0`的数组个数。前缀和+异或，当`xor == pre`时，即说明存在有效数组。

```go
func beautifulSubarrays(nums []int) int64 {
    counter := make(map[int]int)
    counter[0] = 1
    answer := 0
    xor := 0
    for _,num := range(nums){
        xor^=num
        answer+=counter[xor]
        counter[xor]+=1
    }
    return int64(answer)
}
```
