# Two Pointers

## [Valid Palindrome](https://leetcode.com/problems/valid-palindrome)

A: 双指针，处理特殊情况（大小写、特殊字符），判断并移动指针。

```cpp
bool isPalindrome(string s) {
    for (int start = 0, end = s.size() - 1; start < end; start++, end--) { // Move 2 pointers from each end until they collide
        while (!isalnum(s[start]) && start < end) start++; // Increment left pointer if not alphanumeric
        while (!isalnum(s[end]) && start < end) end--; // Decrement right pointer if no alphanumeric
        if (start >= end)   return true;
        if (toupper(s[start]) != toupper(s[end])) return false; // Exit and return error if not match
    }
    
    return true;
}
```

## [3Sum](https://leetcode.com/problems/3sum)

A: 排列，固定第一个数，双指针，跳过相同数字剪枝。

```cpp
class Solution {
public:
    vector<vector<int> > threeSum(vector<int> &num) {
        vector<vector<int> > res;
        std::sort(num.begin(), num.end());

        for (int i = 0; i < num.size(); i++) {

            int target = -num[i];
            int front = i + 1;
            int back = num.size() - 1;

            while (front < back) {

                int sum = num[front] + num[back];

                // Finding answer which start from number num[i]
                if (sum < target)
                    front++;

                else if (sum > target)
                    back--;

                else {
                    vector<int> triplet = {num[i], num[front], num[back]};
                    res.push_back(triplet);

                    // Processing duplicates of Number 2
                    // Rolling the front pointer to the next different number forwards
                    while (front < back && num[front] == triplet[1]) front++;

                    // Processing duplicates of Number 3
                    // Rolling the back pointer to the next different number backwards
                    while (front < back && num[back] == triplet[2]) back--;
                }

            }

            // Processing duplicates of Number 1
            while (i + 1 < num.size() && num[i + 1] == num[i]) 
                i++;

        }

        return res;

    }
};

```

```go
func threeSum(nums []int) [][]int {
	sort.Ints(nums)
	res := [][]int{}
	// 找出a + b + c = 0
	// a = nums[i], b = nums[left], c = nums[right]
	for i := 0; i < len(nums)-2; i++ {
		// 排序之后如果第一个元素已经大于零，那么无论如何组合都不可能凑成三元组，直接返回结果就可以了
		n1 := nums[i]
		if n1 > 0 {
			break
		}
		// 去重a
		if i > 0 && n1 == nums[i-1] {
			continue
		}
		l, r := i+1, len(nums)-1
		for l < r {
			n2, n3 := nums[l], nums[r]
			if n1+n2+n3 == 0 {
				res = append(res, []int{n1, n2, n3})
				// 去重逻辑应该放在找到一个三元组之后，对b 和 c去重
				for l < r && nums[l] == n2 {
					l++
				}
				for l < r && nums[r] == n3 {
					r--
				}
			} else if n1+n2+n3 < 0 {
				l++
			} else {
				r--
			}
		}
	}
	return res
}
```

## [Container With Most Water](https://leetcode.com/problems/container-with-most-water)

A: 双指针，高度较小一侧的指针滑动直至高度增加。

```cpp
class Solution {
public:
    int maxArea(vector<int>& height) {
        int i = 0;
        int j = height.size() - 1;
        
        int curr = 0;
        int result = 0;
        
        while (i < j) {
            curr = (j - i) * min(height[i], height[j]);
            result = max(result, curr);
            
            if (height[i] <= height[j]) {
                i++;
            } else {
                j--;
            }
        }
        
        return result;
    }
};
```

## [Two Sum II Input Array Is Sorted](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted)

A: 双指针，用`sum`和`target`的大小关系判断滑动方向。

```cpp
class Solution {
public:
    vector<int> twoSum(vector<int>& numbers, int target) {
        int len = numbers.size(), left = 0, right = len - 1;
        int sum = numbers[left] + numbers[right];
        while (sum != target) {
            if (sum < target)   left++;
            if (sum > target)   right--;
            sum = numbers[left] + numbers[right];
        }
        return vector<int> {left + 1, right + 1};
    }
};
```

## [Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water)

A: `i`处的储水量为`min(maxLeft, maxRight) - height[i]`，利用双指针维护`maxLeft`和`maxRight`，每次只更新两者中的较小值。

```cpp
class Solution {
public:
    int trap(vector<int>& height) {
        int i = 0;
        int j = height.size() - 1;
        
        int maxLeft = height[i];
        int maxRight = height[j];
        
        int result = 0;
        
        while (i < j) {
            if (maxLeft <= maxRight) {
                i++;
                maxLeft = max(maxLeft, height[i]);
                result += maxLeft - height[i];
            } else {
                j--;
                maxRight = max(maxRight, height[j]);
                result += maxRight - height[j];
            }
        }
        
        return result;
    }
};
```

## [4Sum](https://leetcode.com/problems/sum)

A: 排序，固定两数变为2Sum。

```cpp
class Solution {
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
        sort(nums.begin(), nums.end());
        vector<vector<int>> ans;
        int n = nums.size();
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                int l = j + 1, r = n - 1;
                long remain = long(target) - long(nums[i]) - long(nums[j]);
                while (l < r) {
                    if (nums[l] + nums[r] == remain) {
                        ans.push_back({nums[i], nums[j], nums[l], nums[r]});
                        ++l; --r;
                        while (l < r && nums[l-1] == nums[l]) ++l; // Skip duplicate nums[l]
                    } else if (nums[l] + nums[r] > remain) {
                        --r;
                    } else {
                        ++l;
                    }
                }
                while (j+1 < n && nums[j] == nums[j+1]) ++j; // Skip duplicate nums[j]
            }
            while (i+1 < n && nums[i] == nums[i+1]) ++i; // Skip duplicate nums[i]
        }
        return ans;
    }
};
```

```go
func fourSum(nums []int, target int) [][]int {
    ans := [][]int{}
    sort.Ints(nums)
    for i := 0; i < len(nums) - 3; i++ {
        if (nums[i] > target && nums[i] >= 0) {
            break
        }
        if i > 0 && nums[i] == nums[i - 1] {
            continue
        }
        for j := i + 1; j < len(nums) - 2; j++ {
            if j > i + 1 && nums[j] == nums[j - 1] {
                continue
            }
            left, right := j + 1, len(nums) - 1
            for left < right {
                nLeft, nRight := nums[left], nums[right]
                sum := nums[i] + nums[j] + nLeft + nRight
                if sum == target {
                    quad := []int{nums[i], nums[j], nLeft, nRight}
                    ans = append(ans, quad)
                    for left < right && nums[left] == nLeft {
                        left++
                    }
                    for left < right && nums[right] == nRight {
                        right--
                    }
                } else if sum > target {
                    right--
                } else {
                    left++
                }
            }
        }
    }
    return ans
}
```

## [Number of Subsequences That Satisfy The Given Sum Condition](https://leetcode.com/problems/number-of-subsequences-that-satisfy-the-given-sum-condition)

A: 2Sum，i，j之间符合要求的个数为组合问题。

1000000007：最小的十位质数，取模可避免超限

```cpp
    int numSubseq(vector<int>& A, int target) {
        sort(A.begin(), A.end());
        int res = 0, n = A.size(), l = 0, r = n - 1, mod = 1e9 + 7;
        vector<int> pows(n, 1);
        for (int i = 1 ; i < n ; ++i)
            pows[i] = pows[i - 1] * 2 % mod; // % mod避免超限
        while (l <= r) {
            if (A[l] + A[r] > target) {
                r--;
            } else {
                res = (res + pows[r - l++]) % mod;
            }
        }
        return res;
    }
```

## [Rotate Array](https://leetcode.com/problems/rotate-array)

A: [详解](https://leetcode.com/problems/rotate-array/solutions/54277/summary-of-c-solutions/?orderBy=most_votes)

```cpp
class Solution 
{
public:
    void rotate(int nums[], int n, int k) 
    {
        if ((n == 0) || (k <= 0))
        {
            return;
        }
        
        int cntRotated = 0;
        int start = 0;
        int curr = 0;
        int numToBeRotated = nums[0];
        int tmp = 0;
        // Keep rotating the elements until we have rotated n 
        // different elements.
        while (cntRotated < n)
        {
            do
            {
                tmp = nums[(curr + k)%n];
                nums[(curr+k)%n] = numToBeRotated;
                numToBeRotated = tmp;
                curr = (curr + k)%n;
                cntRotated++;
            } while (curr != start);
            // Stop rotating the elements when we finish one cycle, 
            // i.e., we return to start.
            
            // Move to next element to start a new cycle.
            start++;
            curr = start;
            numToBeRotated = nums[curr];
        }
    }
};
```

## [Array With Elements Not Equal to Average of Neighbors](https://leetcode.com/problems/array-with-elements-not-equal-to-average-of-neighbors)

A: 排序，每三个改为如下顺序：小、大、中。

```cpp
class Solution {
public:
    vector<int> rearrangeArray(vector<int>& nums) {
        int len = nums.size();
        sort(nums.begin(), nums.end());
        for (int i = 1; i < len - 1; i += 2) {
            swap(nums[i], nums[i + 1]);
        }
        return nums;
    }
};
```

## [Boats to Save People](https://leetcode.com/problems/boats-to-save-people)

A: 双指针法确定乘船安排，优先安排小体重。

```cpp
class Solution {
public:
    int numRescueBoats(vector<int>& people, int limit) {
        int boatCount = 0;
        sort(people.begin(), people.end());
        
        int left = 0;
        int right = people.size() - 1;
        
        while(left <= right){
            int sum = people[left] + people[right];
            if(sum <= limit){
                boatCount++;
                left++;
                right--;
            }
            else{
                boatCount++;
                right--;
            }
        }
        return boatCount;
    }
};
```

## [Valid Palindrome II](https://leetcode.com/problems/valid-palindrome-ii)

A: 出现不符合情况，则分两种情况继续判断回文（递归的空间复杂度较高）。

```cpp
    bool validPalindrome(string s) {
        for (int i = 0, j = s.size() - 1; i < j; i++, j--)
            if (s[i] != s[j]) {
                int i1 = i, j1 = j - 1, i2 = i + 1, j2 = j;
                while (i1 < j1 && s[i1] == s[j1]) {i1++; j1--;};
                while (i2 < j2 && s[i2] == s[j2]) {i2++; j2--;};
                return i1 >= j1 || i2 >= j2;
            }
        return true;
    }
```

## [Minimum Difference Between Highest And Lowest of K Scores](https://leetcode.com/problems/minimum-difference-between-highest-and-lowest-of-k-scores)

A: 排序，比较差距k位两元素的最小差值。

```cpp
class Solution {
public:
    int minimumDifference(vector<int>& nums, int k) {
        sort(nums.begin(), nums.end());
        int res = nums[k-1] - nums[0];
        for (int i = k; i < nums.size(); i++) res = min(res, nums[i] - nums[i-k+1]);
        return res;
    }
};
```

## [Reverse String](https://leetcode.com/problems/reverse-string)

A: 双指针依次交换。

```cpp
class Solution {
public:
    void reverseString(vector<char>& s) {
        int i = 0, j = s.size() - 1;
        while (i < j) {
            swap(s[i], s[j]);
            i++, j--;
        }
    }
};
```

## [Merge Sorted Array](https://leetcode.com/problems/merge-sorted-array)

A: 从后往前比较赋值，如果nums2先消耗完，则nums1的元素不用再考虑位置。

```cpp
class Solution {
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        int i=m-1 , j=n-1 ;
        while (i>=0 && j>=0 ) {
            if (nums1[i]>=nums2[j]) {
                nums1[i+j+1]=nums1[i] ;
                i-- ;
            }else{
                nums1[i+j+1]=nums2[j];
                j-- ;
            }
        }
        while(j>=0){
            nums1[j]=nums2[j];
            j-- ;
        }      
    }
};
```

## [Move Zeroes](https://leetcode.com/problems/move-zeroes)

A: 先遍历一遍将所有非0元素前置，然后将后置位全部归0。

```cpp
class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        int j = 0;
        // move all the nonzero elements advance
        for (int i = 0; i < nums.size(); i++) {
            if (nums[i] != 0) {
                nums[j++] = nums[i];
            }
        }
        for (;j < nums.size(); j++) {
            nums[j] = 0;
        }
    }
};
```

## [Remove Duplicates from Sorted Array](https://leetcode.com/problems/remove-duplicates-from-sorted-array)

A: 遍历找到不重复元素前置，得到元素个数。

```cpp
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        int i = 0, len = nums.size();
        int pre = -101;
        for (int j = 0; j < len; j++) {
            if (nums[j] != pre) {
                nums[i++] = nums[j];
                pre = nums[j];
            }
        }
        return i;
    }
};
```

## [Squares of a Sorted Array](https://leetcode.com/problems/squares-of-a-sorted-array)

A: 双指针，指向数组两端，比较绝对值，从后往前填充答案。

```go
func sortedSquares(nums []int) []int {
    ans := make([]int, len(nums))
    i, j, cur := 0, len(nums) - 1, len(nums) - 1
    for cur >= 0 {
        if abs(nums[i]) > abs(nums[j]) {
            ans[cur] = nums[i] * nums[i]
            i++
        } else {
            ans[cur] = nums[j] * nums[j]
            j--
        }
        cur--
    }
    return ans
}

func abs(n int) int {
    if n >= 0 {
        return n
    } else {
        return -n
    }
}
```
