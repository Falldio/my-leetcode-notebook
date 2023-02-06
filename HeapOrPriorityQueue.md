# Heap / Priority Queue

## [Find Median From Data Stream](https://leetcode.com/problems/find-median-from-data-stream)

A: 维护最大堆（小数堆）、最小堆（大数堆）。

```cpp
class MedianFinder {
public:
    MedianFinder() {
        
    }
    
    void addNum(int num) {
        if (lower.empty()) {
            lower.push(num);
            return;
        }
        
        // 保持小数堆大数堆元素数目大致相同
        if (lower.size() > higher.size()) {
            if (lower.top() > num) {
                higher.push(lower.top());
                lower.pop();
                lower.push(num);
            } else {
                higher.push(num);
            }
        } else {
            if (num > higher.top()) {
                lower.push(higher.top());
                higher.pop();
                higher.push(num);
            } else {
                lower.push(num);
            }
        }
    }
    
    double findMedian() {
        double result = 0.0;
        
        if (lower.size() == higher.size()) {
            result = lower.top() + (higher.top() - lower.top()) / 2.0;
        } else {
            if (lower.size() > higher.size()) {
                result = lower.top();
            } else {
                result = higher.top();
            }
        }
        
        return result;
    }
private:
    priority_queue<int> lower;
    priority_queue<int, vector<int>, greater<int>> higher;
};

/**
 * Your MedianFinder object will be instantiated and called as such:
 * MedianFinder* obj = new MedianFinder();
 * obj->addNum(num);
 * double param_2 = obj->findMedian();
 */
```

## [Single-Threaded CPU](https://leetcode.com/problems/single-threaded-cpu)

A: 堆排序，索引、入队时间和处理时间均存入堆（嵌套`pair`）。

```cpp
class Solution {
public:
    vector<int> getOrder(vector<vector<int>>& tasks) {
        using pp = pair<int,pair<int,int>>;
        vector<pp> vtr;
        int n = tasks.size();
        for(int i = 0; i < n; ++i){
            vtr.push_back({tasks[i][0],{tasks[i][1],i}});
        }
        sort(vtr.begin(),vtr.end());
        priority_queue<pp, vector<pp>, greater<pp>> pq;
        vector<int> res;
        int i = 0;
        long long curr_time = vtr[i].first;
        while(i < n|| !pq.empty()){
            while(i < n && curr_time >= vtr[i].first){
                pq.push({vtr[i].second.first,{vtr[i].second.second,vtr[i].first}});
                ++i;
            }
            curr_time = curr_time += pq.top().first;
            res.push_back(pq.top().second.first);
            pq.pop();
            if( i < n && curr_time < vtr[i].first && pq.empty())
                curr_time = vtr[i].first;
                
        }
        return res;
    }
};
```

## [Kth Largest Element In a Stream](https://leetcode.com/problems/kth-largest-element-in-a-stream)

A: 维护长度为k的最小堆。

```cpp
class KthLargest {
public:
    KthLargest(int k, vector<int>& nums) {
        this->k = k;
        for (int i = 0; i < nums.size(); i++) {
            pq.push(nums[i]);
        }
        while (pq.size() > this->k) {
            pq.pop();
        }
    }
    
    int add(int val) {
        pq.push(val);
        if (pq.size() > k) {
            pq.pop();
        }
        return pq.top();
    }
private:
    int k;
    priority_queue<int, vector<int>, greater<int>> pq; // 最小堆greater
};
```

## [Last Stone Weight](https://leetcode.com/problems/last-stone-weight)

A: 每次取最大堆前两个元素合并再放回堆中。

```cpp
class Solution {
public:
    int lastStoneWeight(vector<int>& stones) {
        priority_queue<int> pq(stones.begin(), stones.end());

        while (pq.size() > 1) {
            int x = pq.top();
            pq.pop();
            int y = pq.top();
            pq.pop();
            if (x != y) {
                pq.push(abs(x - y));
            }
        }
        if (pq.size() == 0) return 0;
        return pq.top();
    }
};
```

## [K Closest Points to Origin](https://leetcode.com/problems/k-closest-points-to-origin)

A: `partial_sort`或者`nth_element`。

`partial_sort`: 将 [first, last) 范围内最小（或最大）的 middle-first 个元素移动到 [first, middle) 区域中，并对这部分元素做升序（或降序）排序。

`nth_element`: 从某个序列中找到第 n 小的元素 K，并将 K 移动到序列中第 n 的位置处。不仅如此，整个序列经过 nth_element() 函数处理后，所有位于 K 之前的元素都比 K 小，所有位于 K 之后的元素都比 K 大。

```cpp
class Solution {
public:
    vector<vector<int>> kClosest(vector<vector<int>>& points, int K) {
        partial_sort(points.begin(), points.begin() + K, points.end(), [](vector<int>& p, vector<int>& q) {
            return p[0] * p[0] + p[1] * p[1] < q[0] * q[0] + q[1] * q[1]; // less 升序
        });
        return vector<vector<int>>(points.begin(), points.begin() + K);
    }
};
```

```cpp
class Solution {
public:
    vector<vector<int>> kClosest(vector<vector<int>>& points, int K) {
        nth_element(points.begin(), points.begin() + K - 1, points.end(), [](vector<int>& p, vector<int>& q) {
            return p[0] * p[0] + p[1] * p[1] < q[0] * q[0] + q[1] * q[1];
        });
        return vector<vector<int>>(points.begin(), points.begin() + K);
    }
};
```

A: 最小堆。

```cpp
class Solution {
public:
    vector<vector<int>> kClosest(vector<vector<int>>& points, int K) {
        priority_queue<vector<int>, vector<vector<int>>, compare> pq(points.begin(), points.end());
        vector<vector<int>> ans;
        for (int i = 0; i < K; i++) {
            ans.push_back(pq.top());
            pq.pop();
        }
        return ans;
    }
private:
    struct compare {
        bool operator()(vector<int>& p, vector<int>& q) {
            return p[0] * p[0] + p[1] * p[1] > q[0] * q[0] + q[1] * q[1];
        }
    };
};
```

## [Kth Largest Element In An Array](https://leetcode.com/problems/kth-largest-element-in-an-array)

A: 参照上一个问题。

```cpp
class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        priority_queue<int> pq(nums.begin(), nums.end());
        for (int i = 0; i < k - 1; i++) {
            pq.pop();
        }
        return pq.top();
    }
};
```

## [Task Scheduler](https://leetcode.com/problems/task-scheduler)

A: 优先安排最高频次任务。

```cpp
class Solution {
public:
    int leastInterval(vector<char>& tasks, int n) {
        unordered_map<char,int>mp;
        int count = 0;
        // 收集任务总数信息
        for(auto e : tasks){
            mp[e]++;
            count = max(count, mp[e]);
        }
        
        int ans = (count-1)*(n+1); // 优先安排最高频次任务序列，中间插入其他类型任务
        for(auto e : mp) if(e.second == count) ans++; // 多余任务被插入到序列后方
        return max((int)tasks.size(), ans);
    }
};
```

## [Design Twitter](https://leetcode.com/problems/design-twitter)

A: 按照调用顺序组织所有tweets，查询时后向遍历。

```cpp
class Twitter {
public:
    Twitter() {
        
    }
    
    void postTweet(int userId, int tweetId) {
        posts.push_back({userId, tweetId});
    }
    
    vector<int> getNewsFeed(int userId) {
        // 10 tweets
        int count = 10;
        vector<int> result;
        
        // since postTweet pushes to the back, looping from back gets most recent
        for (int i = posts.size() - 1; i >= 0; i--) {
            if (count == 0) {
                break;
            }
            
            int followingId = posts[i].first;
            int tweetId = posts[i].second;
            unordered_set<int> following = followMap[userId];
            // add to result if they're following them or it's a tweet from themself
            if (following.find(followingId) != following.end() || followingId == userId) {
                result.push_back(tweetId);
                count--;
            }
        }
        
        return result;
    }
    
    void follow(int followerId, int followeeId) {
        followMap[followerId].insert(followeeId);
    }
    
    void unfollow(int followerId, int followeeId) {
        followMap[followerId].erase(followeeId);
    }
private:
    // pairs: [user, tweet]
    vector<pair<int, int>> posts;
    // hash map: {user -> people they follow}
    unordered_map<int, unordered_set<int>> followMap;
};
```

## [Seat Reservation Manager](https://leetcode.com/problems/seat-reservation-manager)

A: 最小堆。

```cpp
class SeatManager {
public:
    SeatManager(int n) {
        for (int i = 1; i <= n; i++) {
            pq.push(i);
        }
    }
    
    int reserve() {
        int ans = pq.top();
        pq.pop();
        return ans;
    }
    
    void unreserve(int seatNumber) {
        pq.push(seatNumber);
    }
private:
    priority_queue<int, vector<int>, greater<int>> pq;
};

/**
 * Your SeatManager object will be instantiated and called as such:
 * SeatManager* obj = new SeatManager(n);
 * int param_1 = obj->reserve();
 * obj->unreserve(seatNumber);
 */
```

## [Process Tasks Using Servers](https://leetcode.com/problems/process-tasks-using-servers)

A: 维持两个最小堆（空闲服务器和工作服务器）。

```cpp
class Solution {
public:
    vector<int> assignTasks(vector<int>& servers, vector<int>& tasks) {
        int n = servers.size();
        int m = tasks.size();
        
        // store {weight, server_index}
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> free_server_pq;
        // store {end_time, server_index}, use long instead of int because the time maybe overflow
        priority_queue<pair<long, long>, vector<pair<long, long>>, greater<pair<long, long>>> busy_server_pq;
        vector<int> ans(m);
        
        for (int i = 0; i < n; ++i) {
            free_server_pq.push({servers[i], i});
        }
        
        long time = 0;
        for (int i = 0; i < m; ++i) {
            time = max(static_cast<long>(i), time);
            if (free_server_pq.empty() && busy_server_pq.top().first > time) {
                time = busy_server_pq.top().first;
            }
            
            while (!busy_server_pq.empty() && busy_server_pq.top().first <= time) {
                auto &info = busy_server_pq.top();
                int server_idx = static_cast<int>(info.second);
                free_server_pq.push({servers[server_idx], server_idx});
                busy_server_pq.pop();
            }
            
            auto &info = free_server_pq.top();
            busy_server_pq.push({time + tasks[i], info.second});
            ans[i] = info.second;
            free_server_pq.pop();
        }
        
        return ans;
    } 
};
```

## [Find The Kth Largest Integer In The Array](https://leetcode.com/problems/find-the-kth-largest-integer-in-the-array)

A: 最大堆。

```cpp
class NumStrMaxComparator {
public:
    bool operator() (string &a, string &b) {
        if (a.size() != b.size()) return a.size() < b.size();
        return  a < b;
    }
};

class Solution {
public:
    string kthLargestNumber(vector<string>& nums, int k) {
        make_heap(nums.begin(), nums.end(), NumStrMaxComparator()); // heapify cost O(N)
        while (k-- > 1) {
            pop_heap(nums.begin(), nums.end(), NumStrMaxComparator());
            nums.pop_back();
        }
        return nums.front();
    }
};
```

```go
func kthLargestNumber(nums []string, k int) string {
    var heap []string
    
    for i := 0; i < len(nums); i++ {
        heap = append(heap, nums[i])
        heapUp(heap, len(heap)-1)
        
        if len(heap) > k {
            heap[0] = heap[len(heap)-1]
            heap = heap[:len(heap)-1]
            heapDown(heap, 0, len(heap)-1)
        }
    }
    
    return heap[0]
}

// 向上冒泡
func heapUp(heap []string, position int) {
    parent := (position-1)/2

    if parent >= 0 && compare(heap[position], heap[parent]) == -1 {
        heap[parent], heap[position] = heap[position], heap[parent]
        heapUp(heap, parent)
    }
}

// 向下冒泡
func heapDown(heap []string, position, limit int) {
    l, r := 2*position+1, 2*position+2
    smaller := position
    
    if l <= limit && compare(heap[l], heap[smaller]) == -1 {
        smaller = l
    }
    
    if r <= limit && compare(heap[r], heap[smaller]) == -1 {
        smaller = r
    }
    
    if smaller != position {
        heap[smaller], heap[position] = heap[position], heap[smaller]
        heapDown(heap, smaller, limit)
    }
}

func compare(s1, s2 string) int {
    if len(s1) > len(s2) {
        return 1
    }
    
    if len(s1) < len(s2) {
        return -1
    }
    
    res := 1
    
    for i := 0; i < len(s1); i++ {
        if s1[i] > s2[i] {
            return 1
        } else if s1[i] < s2[i] {
            return -1
        }
    }
    
    return res
}
```

## [Reorganize String](https://leetcode.com/problems/reorganize-string)

A: 首先安插最频繁的字母，然后安插剩余字符。

```cpp
class Solution {
public:
    string reorganizeString(string S) {
        vector<int> cnt(26);
        int mostFreq = 0, i = 0;
    
        for(char c : S)
            if(++cnt[c - 'a'] > cnt[mostFreq])
                mostFreq = (c - 'a');
    
        if(2 * cnt[mostFreq] - 1 > S.size()) return "";
    
        while(cnt[mostFreq]) {
            S[i] = ('a' + mostFreq);
            i += 2;
            cnt[mostFreq]--;
        }
    
        for(int j = 0; j < 26; j++) {
            while(cnt[j]) {
                if(i >= S.size()) i = 1; // i从已添加位置处继续，防止aabbcc或类似情况无法插入
                S[i] = ('a' + j);
                cnt[j]--;
                i += 2;
            }
        }
    
        return S;
    }
};
```

## [Longest Happy String](https://leetcode.com/problems/longest-happy-string)

A: 最大堆，取堆顶两个元素，依次填充字符串。

```cpp
class Solution {
public:
    string longestDiverseString(int a, int b, int c) {
        //using max heap
        priority_queue<pair<int,char>>pq;
        if(a)
        pq.push({a,'a'});
        if(b)
        pq.push({b,'b'});
        if(c)
        pq.push({c,'c'});
        string ans="";
        while(pq.size()>1){
            pair<int,char>one = pq.top();pq.pop();
            pair<int,char>two = pq.top();pq.pop();
            if(one.first>=2){
                ans+=one.second;
                ans+=one.second;
                one.first-=2;
            }
            else{
                ans+=one.second;
                one.first-=1;
            }
            if(two.first>=2 && two.first>=one.first){
                ans+=two.second;
                ans+=two.second;
                two.first-=2;
            }
            else{
                ans+=two.second;
                two.first-=1;
            }
            if(one.first>0)
                pq.push(one);
            if(two.first>0)
                pq.push(two);
        }
        if(pq.empty())
            return ans;
        if(pq.top().first>=2){
            ans+=pq.top().second;
            ans+=pq.top().second;
        }
        else{
            ans+=pq.top().second;
        }
        return ans;
        
    }
};
```

## [Car Pooling](https://leetcode.com/problems/car-pooling)

A: 统计每一个站点的capacity变化情况，最后查看capacity是否取值正常。

```cpp
bool carPooling(vector<vector<int>>& trips, int capacity) {
  int stops[1001] = {};
  for (auto t : trips) stops[t[1]] += t[0], stops[t[2]] -= t[0];
  for (auto i = 0; capacity >= 0 && i < 1001; ++i) capacity -= stops[i];
  return capacity >= 0;
}
```

## [Maximum Performance of a Team](https://leetcode.com/problems/maximum-performance-of-a-team)

A: 排序efficiency，每当满足k时去掉speed最小的元素。

[详解](https://leetcode.com/problems/maximum-performance-of-a-team/solutions/539687/java-c-python-priority-queue/?orderBy=most_votes)

```cpp
class Solution {
public:
    int maxPerformance(int n, vector<int>& speed, vector<int>& efficiency, int k) {
        vector<pair<int, int>> ess(n);
        for (int i = 0; i < n; ++i)
            ess[i] = {efficiency[i], speed[i]};
        sort(rbegin(ess), rend(ess));
        long sumS = 0, res = 0;
        priority_queue <int, vector<int>, greater<int>> pq; //min heap
        for(auto& [e, s]: ess){
            pq.emplace(s);
            sumS += s;
            if (pq.size() > k) {
                sumS -= pq.top();
                pq.pop();
            }
            res = max(res, sumS * e);
        }
        return res % (int)(1e9+7);
    }
};
```

## [Minimum Cost to Hire K Workers](https://leetcode.com/problems/minimum-cost-to-hire-k-workers)

A: 按照工资比例排序，按照最高的工资比例来支付所有工资，超过k时去掉quality最高的员工以降低总工资。

```cpp
class Solution {
public:
    double mincostToHireWorkers(vector<int> q, vector<int> w, int K) {
        vector<vector<double>> workers;
        for (int i = 0; i < q.size(); ++i)
            // w / q: waging ratio
            workers.push_back({(double)(w[i]) / q[i], (double)q[i]});
        sort(workers.begin(), workers.end());
        double res = DBL_MAX, qsum = 0;
        priority_queue<int> pq;
        for (auto worker: workers) {
            qsum += worker[1], pq.push(worker[1]);
            if (pq.size() > K) qsum -= pq.top(), pq.pop();
            if (pq.size() == K) res = min(res, qsum * worker[0]);
        }
        return res;
    }
};
```
