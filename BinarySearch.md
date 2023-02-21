# Binary Search

**二分查找**：O(logn)

## [Search In Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array)

A: 判断mid在旋转数组的哪一个顺序部分，利用左右边界的大小判断比较窗口边界。

```cpp
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int low = 0;
        int high = nums.size() - 1;
        
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (nums[mid] == target) {
                return mid;
            }
            if (nums[low] <= nums[mid]) {
                // mid在左侧顺序数组
                if (nums[low] <= target && target <= nums[mid]) {
                    high = mid - 1;
                } else {
                    // nums[low] > target || target > nums[mid]
                    low = mid + 1;
                }
            } else {
                // mid在右侧顺序数组
                if (nums[mid] <= target && target <= nums[high]) {
                    low = mid + 1;
                } else {
                    // nums[mid] > target || target > nums[high]
                    high = mid - 1;
                }
            }
        }
        
        return -1;
    }
};
```

## [Find Minimum In Rotated Sorted Array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array)

A: 判断mid所属数组部分，调整检索范围，直到检索窗口位于顺序数组。

```cpp
class Solution {
public:
    int findMin(vector<int>& nums) {
        int l = 0, r = nums.size() - 1;
        int min = INT_MAX;

        while (l <= r) {
            int m = l + (r - l) / 2;
            if (nums[l] <= nums[r]) {
                min = std::min(min, nums[l]);
                break;
            }
            min = std::min(min, nums[m]);
            if (nums[m] >= nums[l]) {
                l = m + 1;
            } else {
                r = m - 1;
            }
        }

        return min;
    }
};
```

## [Binary Search](https://leetcode.com/problems/binary-search)

A: 简单二分查找。

```cpp
int search(vector<int>& nums, int target) {
        int n = nums.size()-1;
        int low = 0, high = n;
        while( low <= high){
            int mid = low + (high-low)/2;
            if (nums[mid] == target) return mid;
            else if (nums[mid] > target) high = mid -1;
            else low = mid + 1;
        }
        return -1;
}
```

```go
func search(nums []int, target int) int {
    left, right := 0, len(nums) - 1

    for left <= right {
        mid := left + (right - left) / 2
        if nums[mid] == target {
            return mid
        } else if nums[mid] < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }

    return -1
}
```

## [Search a 2D Matrix](https://leetcode.com/problems/search-a-2d-matrix)

A: 将一维索引转换为二维索引。

```cpp
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int row = matrix.size();
        int col = matrix[0].size();
        int l = 0, r = row * col - 1;
        while (l <= r) {
            int numLeft = matrix[l / col][l % col];
            int numRight = matrix[r / col][r % col];
            int mid = l + (r - l) / 2;
            int numMid = matrix[mid / col][mid % col];
            if (numMid == target)   return true;
            if (numMid < target)    l = mid + 1;
            else    r = mid - 1;
        }
        return false;
    }
};
```

## [Koko Eating Bananas](https://leetcode.com/problems/koko-eating-bananas)

A: 搜索范围为速度：`[1, max(piles)]`或`[1, 10^9]`（根据限制条件），找到满足条件的最小速度。

```cpp
class Solution {
public:
    int minEatingSpeed(vector<int>& piles, int H) {
        int low = 1, high = 1000000000, k = 0;
        while (low <= high) {
            k = (low + high) / 2;
            double h = 0;
            for (int i = 0; i < piles.size(); i ++) 
                h += ceil(1.0 * piles[i] / k);
            if (h > H)
                low = k + 1;
            else
                high = k - 1;
        }
        return low;
    }
};
```

## [Time Based Key Value Store](https://leetcode.com/problems/time-based-key-value-store)

A: 使用`upper_bound`查找上界。

`map`底层为红黑树，`unordered_map`底层为哈希表，vector。

**TODO**: `upper_bound`和`lower_bound`实现。

```cpp
class TimeMap {
public:
    unordered_map<string, map<int, string>> m;
    TimeMap() {
        
    }
    
    void set(string key, string value, int timestamp) {
        m[key][timestamp] = value;
    }
    
    string get(string key, int timestamp) {
        auto it = m[key].upper_bound(timestamp);
        return it == m[key].begin() ? "" : prev(it)->second;
    }
};

/**
 * Your TimeMap object will be instantiated and called as such:
 * TimeMap* obj = new TimeMap();
 * obj->set(key,value,timestamp);
 * string param_2 = obj->get(key,timestamp);
 */
 ```

## [Median of Two Sorted Arrays](https://leetcode.com/problems/median-of-two-sorted-arrays)

A: 在两个数组中分别标定左半部分，通过它们与另一个数组右半部分最小元素的大小关系来调整边界。

```cpp
class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int m = nums1.size();
        int n = nums2.size();
        
        if (m > n) {
            return findMedianSortedArrays(nums2, nums1);
        }
        if (m == 0) {
            if (n % 2 == 0){
                return (nums2[n/2 - 1] + nums2[n/2]) / 2.0;
            } else {
                return nums2[n/2];
            }
        }
        
        int total = m + n;
        
        int low = 0;
        int high = m;
        
        double result = 0.0;
        
        while (low <= high) {
            // nums1
            int i = low + (high - low) / 2; // low, i, high
            // nums2
            int j = (total + 1) / 2 - i; // nums2的左半部分
            
            int left1 = (i > 0) ? nums1[i - 1] : INT_MIN;
            int right1 = (i < m) ? nums1[i] : INT_MAX;
            int left2 = (j > 0) ? nums2[j - 1] : INT_MIN;
            int right2 = (j < n) ? nums2[j] : INT_MAX;
            
            // partition is correct
            if (left1 <= right2 && left2 <= right1) {
                // even
                if (total % 2 == 0) {
                    result = (max(left1, left2) + min(right1, right2)) / 2.0;
                // odd
                } else {
                    result = max(left1, left2);
                }
                break;
            } else if (left1 > right2) {
                high = i - 1;
            } else {
                low = i + 1;
            }
        }
        
        return result;
    }
};
```

## [Find First And Last Position of Element In Sorted Array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array)

A: 二分查找+左右扩展边界。

```cpp
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1, mid;
        bool flag = false;
        while (left <= right) {
            mid = left + (right - left) / 2;
            if (nums[mid] > target) {
                right = mid - 1;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                flag = true;
                break;
            }
        }
        if (flag) {
            int i = mid, j = mid;
            while (i >= 0 && nums[i] == target) i--;
            while (j <= nums.size() - 1 && nums[j] == target) j++;
            return {i + 1, j - 1};
        }
        return {-1, -1};
    }
};
```

## [Maximum Number of Removable Characters](https://leetcode.com/problems/maximum-number-of-removable-characters)

A: 二分查找removable的索引，找到满足条件的最大索引值。

```cpp
class Solution {
public:
    int maximumRemovals(string s, string p, vector<int>& removable) {
        int left = 0, right = removable.size() - 1;
        int ans = 0;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (isValid(s, p, removable, mid)) {
                left = mid + 1;
                ans = max(ans, mid + 1);
            } else {
                right = mid - 1;
            }
        }
        return ans;
    }
private:
    bool isValid(string s, string &p, vector<int>& removable, int mid) {
        int len1 = s.size(), len2 = p.size();
        for (int i = 0; i <= mid; i++) {
            s[removable[i]] = '0';
        }
        int j = 0;
        for (int i = 0; i < len1; i++) {
            if (s[i] == '0') continue;
            if (s[i] == p[j]) j++;
        }
        return j == len2;
    }
};
```

## [Search Suggestions System](https://leetcode.com/problems/search-suggestions-system)

A: 对整个字符串进行排序和二分查找。

```cpp
class Solution {
public:
    vector<vector<string>> suggestedProducts(vector<string>& A, string searchWord) {
        auto it = A.begin();
        sort(it, A.end());
        vector<vector<string>> res;
        string cur = "";
        for (char c : searchWord) {
            cur += c;
            vector<string> suggested;
            it = lower_bound(it, A.end(), cur);
            for (int i = 0; i < 3 && it + i != A.end(); i++) {
                string& s = *(it + i);
                if (s.find(cur)) break; // string的find()函数找不到则返回npos，一个cpp特别标志，因为是前缀，所以找到则返回0
                suggested.push_back(s);
            }
            res.push_back(suggested);
        }
        return res;
    }
};
```

## [Split Array Largest Sum](https://leetcode.com/problems/split-array-largest-sum)

A: 答案的取值范围为\[最大值，数组元素之和\]，在该区间寻找最小值。

```cpp
class Solution {
public:
    int splitArray(vector<int>& nums, int m) {
        int l=0,r=0,n=nums.size();
        for(int i=0;i<n;++i) l=max(l,nums[i]), r+=nums[i];
        
        int mid=0,ans=0;
        while(l<=r){
            mid=(l+r)/2;
            int count=0,tempsum=0;
            for(int i=0;i<n;++i){
                // 在取值范围中值处切分数组
                if(tempsum+nums[i]<=mid) tempsum+=nums[i];
                else count++,tempsum=nums[i];
            }
            count++; 
            
            if(count<=m) r=mid-1, ans=mid;
            else l=mid+1;
        }  
        return ans;
    }
};
```

## [Search Insert Position](https://leetcode.com/problems/search-insert-position)

A: 简单二分查找。

```cpp
class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        int i = 0, j = nums.size() - 1;
        while (i <= j) {
            int mid = i + (j - i) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] < target) {
                i = mid + 1;
            } else {
                j = mid - 1;
            }
        }
        return i;
    }
};
```

```go
func searchInsert(nums []int, target int) int {
    i, j := 0, len(nums) - 1
    if target > nums[j] {
        return len(nums)
    }
    if target < nums[i] {
        return 0
    }
    for i <= j {
        mid := i + (j - i) / 2
        if nums[mid] == target {
            return mid
        } else if nums[mid] < target {
            i = mid + 1
        } else {
            j = mid - 1
        }
    }
    if nums[i] > target {
        return i
    } else {
        return i + 1
    }
}
```

## [Guess Number Higher or Lower](https://leetcode.com/problems/guess-number-higher-or-lower)

A: 二分查找，每次都猜测中位数。

```cpp
/** 
 * Forward declaration of guess API.
 * @param  num   your guess
 * @return       -1 if num is higher than the picked number
 *               1 if num is lower than the picked number
 *               otherwise return 0
 * int guess(int num);
 */

class Solution {
public:
    int guessNumber(int n) {
        int i = 1, j = n;
        while (i <= j) {
            int mid = i + (j - i) / 2;
            int res = guess(mid);
            if (res == 0) {
                return mid;
            } else if (res == -1) {
                j = mid - 1;
            } else {
                i = mid + 1;
            }
        }
        return i;
    }
};
```

## [Arranging Coins](https://leetcode.com/problems/arranging-coins)

A: 用二分查找寻找\[1, n]范围内最大的row。

```cpp
class Solution {
public:
    int arrangeCoins(int n) {
        int l = 1, r = n, ans;
        long rows, coinsNeeded;
        while(l <= r) {
            rows = l + ((r-l) >> 1);                            // finding mid of range [l, r]
            coinsNeeded = (rows * (rows + 1)) >> 1;             // coins needed for 'rows' number of row
            if(coinsNeeded <= n) l = rows + 1, ans = rows;      // if available coins are sufficient
            else r = rows - 1;                                  // coins insufficient, eliminate the half greater than rows
        }
        return ans;
    }
};
```

## [Squares of a Sorted Array](https://leetcode.com/problems/squares-of-a-sorted-array)

A: 双指针，比较两侧绝对值，决定答案顺序。

```cpp
class Solution {
public:
    vector<int> sortedSquares(vector<int>& A) {
        vector<int> res(A.size());
        int l = 0, r = A.size() - 1;
        for (int k = A.size() - 1; k >= 0; k--) {
            if (abs(A[r]) > abs(A[l])) res[k] = A[r] * A[r--];
            else res[k] = A[l] * A[l++];
        }
        return res;
    }
};
```

## [Valid Perfect Square](https://leetcode.com/problems/valid-perfect-square)

A: 在\[1, num]区间二分查找。

```cpp
class Solution {
public:
    bool isPerfectSquare(int num) {
        long i = 1, j = num;
        while (i <= j) {
            long mid = (i + j) / 2;
            long sqrt = mid * mid;
            if (sqrt == num) {
                return true;
            } else if (sqrt < num) {
                i = mid + 1;
            } else {
                j = mid - 1;
            }
        }
        return false;
    }
};
```

## [Single Element in a Sorted Array](https://leetcode.com/problems/single-element-in-a-sorted-array)

A: 正常情况下同一个元素出现两次，占据一个奇数位置和一个偶数位置（数组从0开始，因此先偶后奇）。因此对于中间位置，对其奇偶性进行判断：如果是奇数，正常情况下左边元素应该和中间元素相等，如果不等，说明单独元素在左边；如果是偶数，正常情况下右边元素应该和中间元素相等，如果不等，说明单独元素在右边。

```go
func singleNonDuplicate(nums []int) int {
    i, j := 0, len(nums) - 1
    if len(nums) == 1 || nums[0] != nums[1] {
        return nums[0]
    }
    if nums[j] != nums[j - 1] {
        return nums[j]
    }
    for i < j {
        mid := i + (j - i) / 2
        if (mid % 2 == 0 && nums[mid + 1] == nums[mid]) || (mid % 2 != 0 && nums[mid - 1] == nums[mid]) {
            i = mid + 1
        } else {
            j = mid
        }
    }
    return nums[i]
}
```
