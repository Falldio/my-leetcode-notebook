# Intervals

## [Insert Interval](https://leetcode.com/problems/insert-interval)

A: 利用二分查找快速找到插入位置，`equal_range`返回first和last之间等于val的元素区间。

```cpp
class Solution {
public:
    vector<Interval> insert(vector<Interval>& intervals, Interval newInterval) {
        auto compare = [] (const Interval &intv1, const Interval &intv2)
                          { return intv1.end < intv2.start; };
        auto range = equal_range(intervals.begin(), intervals.end(), newInterval, compare);
        auto itr1 = range.first, itr2 = range.second;
        if (itr1 == itr2) {
            intervals.insert(itr1, newInterval);
        } else {
            itr2--;
            itr2->start = min(newInterval.start, itr1->start);
            itr2->end = max(newInterval.end, itr2->end);
            intervals.erase(itr1, itr2);
        }
        return intervals;
    }
};
```

```go
func insert(intervals [][]int, newInterval []int) [][]int {
    
    res := make([][]int, 0)
    
    i := 0
    
    for ; i < len(intervals) && intervals[i][1] < newInterval[0]; i++ {
        res = append(res, intervals[i])
    }
    
    for ; i < len(intervals) && intervals[i][0] <= newInterval[1]; i++ {
        newInterval[0] = min(intervals[i][0], newInterval[0])
        newInterval[1] = max(intervals[i][1], newInterval[1])
    }
    
    res = append(res, newInterval)
    
    for i < len(intervals) {
        res = append(res, intervals[i])
        i++
    }
    
    return res
}

func min(x, y int) int {
    if x < y {
        return x
    }
    return y
}

func max(x, y int) int {
    if x > y {
        return x
    }
    return y
}
```

A: 分三段组织插入顺序。

```cpp
class Solution {
public:
    vector<vector<int>> insert(vector<vector<int>>& intervals, vector<int>& newInterval) {
        int i = 0;
        int n = intervals.size();
        
        vector<vector<int>> result;
        
        while (i < n && intervals[i][1] < newInterval[0]) {
            result.push_back(intervals[i]);
            i++;
        }
        
        while (i < n && intervals[i][0] <= newInterval[1]) {
            newInterval[0] = min(newInterval[0], intervals[i][0]);
            newInterval[1] = max(newInterval[1], intervals[i][1]);
            i++;
        }
        result.push_back(newInterval);
        
        while (i < n) {
            result.push_back(intervals[i]);
            i++;
        }
        
        return result;
    }
};
```

## [Meeting Room](https://www.lintcode.com/problem/920)

A: 按`start`排列再检查是否冲突。

```cpp
class Solution {
public:
    bool canAttendMeetings(vector<vector<int>>& intervals) {
        if (intervals.empty()) {
            return true;
        }
        
        sort(intervals.begin(), intervals.end());
        for (int i = 0; i < intervals.size() - 1; i++) {
            if (intervals[i][1] > intervals[i + 1][0]) {
                return false;
            }
        }
        return true;
    }
};
```

## [Meeting Room](https://www.lintcode.com/problem/919)

A: 双指针，维护`start`和`end`两个数组，以此计算最大同时会议数目。

```python
class Solution:
    """
    @param intervals: an array of meeting time intervals
    @return: the minimum number of conference rooms required
    """

    def minMeetingRooms(self, intervals):
        start = sorted([i[0] for i in intervals])
        end = sorted([i[1] for i in intervals])

        res, count = 0, 0
        s, e = 0, 0
        while s < len(intervals):
            if start[s] < end[e]:
                s += 1
                count += 1
            else:
                e += 1
                count -= 1
            res = max(res, count)
        return res
```

## [Merge Intervals](https://leetcode.com/problems/merge-intervals)

A: 滑动窗口。

```cpp
class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        sort(intervals.begin(), intervals.end());

        int i = 0, j = 1;
        int len = intervals.size();
        vector<vector<int>> ans;

        while (j < len + 1) {
            if (j == len) {
                ans.push_back(intervals[i]);
                return ans;
            }
            if (intervals[i][1] < intervals[j][0]) {
                ans.push_back(intervals[i]);
                i = j;
                j++;
            } else {
                intervals[i][1] = max(intervals[i][1], intervals[j][1]);
                j++;
            }
        }

        return ans;
    }
};
```

## [Non-overlapping Intervals](https://leetcode.com/problems/non-overlapping-intervals)

A: 贪心，出现重叠情况，则舍去end最大的interval。

```cpp
class Solution {
public:
    int eraseOverlapIntervals(vector<vector<int>>& intervals) {
        int ans = 0;
        sort(intervals.begin(), intervals.end());
        int len = intervals.size();
        vector<int> cur = intervals[0];

        for (int i = 1; i < len; i++) {
            if (intervals[i][0] >= cur[1]) {
                cur = intervals[i];
            } else {
                cur[1] = min(cur[1], intervals[i][1]);
                ans++;
            }
        }

        return ans;
    }
};
```

```go
func eraseOverlapIntervals(intervals [][]int) int {
    sort.Slice(intervals, func(i, j int) bool {
        if intervals[i][0] < intervals[j][0] {
            return true
        } else if intervals[i][0] == intervals[j][0] {
            return intervals[i][1] < intervals[j][1]
        } else {
            return false
        }
    })
    ans := 0
    curEnd := intervals[0][1]
    for i := 1; i < len(intervals); i++ {
        if intervals[i][0] < curEnd {
            ans++
            if curEnd > intervals[i][1] {
                curEnd = intervals[i][1]
            }
        } else {
            curEnd = intervals[i][1]
        }
    }
    return ans
}
```

## [Minimum Number of Arrows to Burst Balloons](https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons)

A: 比较intervals的end，以此判断是否有重叠部分。

```cpp
bool cmp(vector<int>& a, vector<int>& b) {return a[1] < b[1];}
class Solution {
public:  
    int findMinArrowShots(vector<vector<int>>& segments) {
        sort(segments.begin(), segments.end(), cmp);
        int ans = 0, arrow = 0;
        for (int i = 0; i < segments.size(); i ++) {
            if (ans == 0 || segments[i][0] > arrow) {
                ans ++;
                arrow = segments[i][1];
            }
        }
        return ans;
    }
};
```

```go
func findMinArrowShots(points [][]int) int {
    sort.Slice(points, func(i, j int) bool {
        if points[i][0] < points[j][0] {
            return true
        } else if points[i][0] == points[j][0] {
            return points[i][1] < points[j][1]
        } else {
            return false
        }
    })
    ans := 1
    curStart, curEnd := points[0][0], points[0][1]
    for i := 1; i < len(points); i++ {
        curStart = points[i][0]
        if points[i][1] < curEnd {
            curEnd = points[i][1]
        }
        if curStart > curEnd {
            ans++
            curEnd = points[i][1]
        }
    }
    return ans
}
```

## [Minimum Interval to Include Each Query](Minimum Interval to Include Each Query)

A: 用优先队列存储{size， right}，对两个参数排序，将满足要求的interval加入最小堆，但不pop，以供后续更大query使用。

```cpp
class Solution {
public:
    vector<int> minInterval(vector<vector<int>>& intervals, vector<int>& queries) {
        vector<int> sortedQueries = queries;
        
        // [size of interval, end of interval]
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
        // {query -> size of interval}
        unordered_map<int, int> m;
        
        // also need only valid intervals so sort by start time & sort queries
        sort(intervals.begin(), intervals.end());
        sort(sortedQueries.begin(), sortedQueries.end());
        
        vector<int> result;
        
        int i = 0;
        for (int j = 0; j < sortedQueries.size(); j++) {
            int query = sortedQueries[j];
            
            while (i < intervals.size() && intervals[i][0] <= query) {
                int left = intervals[i][0];
                int right = intervals[i][1];
                pq.push({right - left + 1, right});
                i++;
            }
            
            while (!pq.empty() && pq.top().second < query) {
                pq.pop();
            }
            
            if (!pq.empty()) {
                m[query] = pq.top().first;
            } else {
                m[query] = -1;
            }
        }
        
        for (int j = 0; j < queries.size(); j++) {
            result.push_back(m[queries[j]]);
        }
        return result;
    }
};
```

## [Remove Covered Intervals](https://leetcode.com/problems/remove-covered-intervals)

A: 按左端点升序排列，左端点相同则按右端点降序排列，确保较长interval排列在前。

```cpp
class Solution {
public:
    int removeCoveredIntervals(vector<vector<int>>& intervals) {
        int ans = intervals.size();
        sort(intervals.begin(), intervals.end(), [](vector<int> &lhs, vector<int> &rhs) {
            if (lhs[0] == rhs[0]) {
                return lhs[1] > rhs[1];
            } else {
                return lhs[0] < rhs[0];
            }
        });
        int start = INT_MIN, end = INT_MIN;
        for (int i = 0; i < intervals.size(); i++) {
            if (intervals[i][0] >= start && intervals[i][1] <= end) {
                ans--;
            } else {
                start = intervals[i][0];
                end = max(end, intervals[i][1]);
            }
        }
        return ans;
    }
};
```

## [Data Stream as Disjoint Intervals](https://leetcode.com/problems/data-stream-as-disjoint-intervals)

A: 二分查找树寻找新数字插入位置。

```cpp
class SummaryRanges {
public:
    SummaryRanges() {
        
    }
    
    void addNum(int value) {
        auto it = _map.lower_bound(value);
        bool merged = false;
        if(it != _map.begin()) {
            auto prev = it;
            --prev;
            if(prev->second + 1 >= value) {
                // 被前一个interval合并
                merged = true;
                prev->second = max(prev->second, value);
            }
        }

        if(it != _map.end()) {
            if(it->first - 1 <= value) {
                // 被后一个interval合并
                if(merged) {
                    auto prev = it;
                    --prev;
                    if(prev->second >= it->first - 1) {
                        // 如果已经被前一个interval合并，则需要检测这两个interval是否需要合并
                        prev->second = max(prev->second, it->second);
                        _map.erase(it);
                    }
                } else {
                    merged = true;
                    if(it->first != value) {
                        pair<int, int> p = *it;
                        p.first = min(p.first, value);
                        it = _map.insert(it, p);
                        ++it;
                        if(it != _map.end())
                            _map.erase(it);
                    }
                }
            }
        }
        if(!merged) // 视为新的插入
            _map.insert(it, {value, value});
    }
    
    vector<vector<int>> getIntervals() {
        vector<vector<int>> intervals;
        for(auto const & p : _map)
            intervals.push_back({p.first, p.second});
        return intervals;
    }

    map<int, int> _map;
};
```
