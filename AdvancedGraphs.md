# Advanced Graphs

## [Alien Dictionary](https://www.lintcode.com/problem/892)

A: 拓扑排序。

```cpp
class Solution {
public:
    string alienOrder(vector<string> &words) {
        unordered_map<char, unordered_set<char>> graph; // 每个字母的邻接表
        unordered_map<char, int> indegree;

        // indegree make all char 0
        for(auto word : words){
            for(auto c : word){
                indegree[c]=0;
            }
        }

        for(int i=0; i<words.size()-1; i++){
            string curr = words[i];
            string next = words[i+1];
            
            bool flag = false; // 标记是否存在字母分支
            int len = min(curr.length(), next.length());
            for(int j=0; j<len; j++){
                char ch1 = curr[j];
                char ch2 = next[j];
                
                // 构建邻接表，即图的结构
                if(ch1 != ch2){
                    unordered_set<char> set;

                    if(graph.find(ch1) != graph.end()){
                        set = graph[ch1];

                        if(set.find(ch2) == set.end()){
                            set.insert(ch2);
                            indegree[ch2]++;
                            graph[ch1] = set;
                        }
                    }
                    else{
                        set.insert(ch2);
                        indegree[ch2]++;
                        graph[ch1] = set;
                    }

                    flag = true; // 两个word不等长，且存在字母分支，即相同位置不同字母
                    break;
                }
                
            }

            if(flag == false and (curr.length() > next.length())) return "";    // 这种情况意味着next是curr的前缀词，即异常输入
        }

        priority_queue<char, vector<char>, greater<char>> q;

        for(auto it : indegree){
            if(it.second == 0){
                q.push(it.first);
            }
        }

        int count=0;
        string ans = "";

        while(q.size()>0){
            // 入度为0即一条路径的出发点
            auto rem = q.top();
            q.pop();

            ans += rem;
            count++;

            if(graph.find(rem) != graph.end()){
                unordered_set<char> nbrs = graph[rem]; // neighbors

                for(auto nbr : nbrs){
                    indegree[nbr]--;
                    if(indegree[nbr] == 0){
                        q.push(nbr);
                    }
                }
            }
        }

        if(count == indegree.size()){
            return ans;
        }
        return "";
    }
};
```

## [Reconstruct Itinerary](https://leetcode.com/problems/reconstruct-itinerary)

A: 构建邻接表，贪心（优先选择词典顺序靠前的城市）+DFS，当找到一个结果后直接返回（决策树中的一个叶子结点），不必再寻找。

```cpp
class Solution {
public:
    vector<string> findItinerary(vector<vector<string>>& tickets) {
        unordered_map<string, multiset<string>> m;
        for (int i = 0; i < tickets.size(); i++) {
            m[tickets[i][0]].insert(tickets[i][1]);
        }
        
        vector<string> result;
        dfs(m, "JFK", result);
        reverse(result.begin(), result.end());
        return result;
    }
private:
    void dfs(unordered_map<string, multiset<string>>& m,
        string airport, vector<string>& result) {
        
        while (!m[airport].empty()) {
            string next = *m[airport].begin();
            m[airport].erase(m[airport].begin());
            dfs(m, next, result);
        }
        
        result.push_back(airport);
    }
};
```

```go
type pair struct {
	target  string
	visited bool
}
type pairs []*pair

func (p pairs) Len() int {
	return len(p)
}
func (p pairs) Swap(i, j int) {
	p[i], p[j] = p[j], p[i]
}
func (p pairs) Less(i, j int) bool {
	return p[i].target < p[j].target
}

func findItinerary(tickets [][]string) []string {
	result := []string{}
	// map[出发机场] pair{目的地,是否被访问过}
	targets := make(map[string]pairs)
	for _, ticket := range tickets {
		if targets[ticket[0]] == nil {
			targets[ticket[0]] = make(pairs, 0)
		}
		targets[ticket[0]] = append(targets[ticket[0]], &pair{target: ticket[1], visited: false})
	}
	for k, _ := range targets {
		sort.Sort(targets[k])
	}
	result = append(result, "JFK")
	var backtracking func() bool
	backtracking = func() bool {
		if len(tickets)+1 == len(result) {
			return true
		}
		// 取出起飞航班对应的目的地
		for _, pair := range targets[result[len(result)-1]] {
			if pair.visited == false {
				result = append(result, pair.target)
				pair.visited = true
				if backtracking() {
					return true
				}
				result = result[:len(result)-1]
				pair.visited = false
			}
		}
		return false
	}

	backtracking()

	return result
}
```

## [Min Cost to Connect All Points](https://leetcode.com/problems/min-cost-to-connect-all-points)

A: Prim's->最小生成树，更新每个结点到生成树的最短距离，每次取离树最近且不在树中的点加入生成树。

```cpp
class Solution {
public:
    int minCostConnectPoints(vector<vector<int>>& points) {
        int n = points.size();
        
        int edgesUsed = 0;
        // track visited nodes
        vector<bool> inMST(n, false);
        vector<int> minDist(n, INT_MAX);
        minDist[0] = 0; // 以便启动算法
        
        int result = 0;
        
        while (edgesUsed < n) {
            int currMinEdge = INT_MAX;
            int currNode = -1;
            
            // greedily pick lowest cost node not in MST
            for (int i = 0; i < n; i++) {
                if (!inMST[i] && currMinEdge > minDist[i]) {
                    currMinEdge = minDist[i];
                    currNode = i;
                }
            }
            
            result += currMinEdge;
            edgesUsed++;
            inMST[currNode] = true;
            
            // update adj nodes of curr node
            for (int i = 0; i < n; i++) {
                int cost = abs(points[currNode][0] - points[i][0])
                    + abs(points[currNode][1] - points[i][1]);
                
                if (!inMST[i] && minDist[i] > cost) {
                    minDist[i] = cost; // 即i到生成树的最短距离
                }
            }
        }
        
        return result;
    }
};
```

## [Network Delay Time](https://leetcode.com/problems/network-delay-time)

A: Dijkstra's->寻找指定结点到其余所有结点的最短路径。

`Dijkstra`: 用优先队列取代BFS中的队列。

```cpp
class Solution {
public:
    int networkDelayTime(vector<vector<int>>& times, int n, int k) {
        vector<pair<int, int>> adj[n + 1];
        for (int i = 0; i < times.size(); i++) {
            int source = times[i][0];
            int dest = times[i][1];
            int time = times[i][2];
            adj[source].push_back({time, dest});
        }
        
        vector<int> signalReceiveTime(n + 1, INT_MAX);
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq; // greater小顶堆
        pq.push({0, k});
        
        // time for start node is 0
        signalReceiveTime[k] = 0;
        
        while (!pq.empty()) {
            int currNodeTime = pq.top().first;
            int currNode = pq.top().second;
            pq.pop();
            
            if (currNodeTime > signalReceiveTime[currNode]) {
                continue;
            }
            
            // send signal to adjacent nodes
            for (int i = 0; i < adj[currNode].size(); i++) {
                pair<int, int> edge = adj[currNode][i];
                int time = edge.first;
                int neighborNode = edge.second;
                
                // fastest signal time for neighborNode so far
                if (signalReceiveTime[neighborNode] > currNodeTime + time) {
                    signalReceiveTime[neighborNode] = currNodeTime + time;
                    pq.push({signalReceiveTime[neighborNode], neighborNode});
                }
            }
        }
        
        int result = INT_MIN;
        for (int i = 1; i <= n; i++) {
            result = max(result, signalReceiveTime[i]);
        }
        
        if (result == INT_MAX) {
            return -1;
        }
        return result;
    }
};
```

## [Swim In Rising Water](https://leetcode.com/problems/swim-in-rising-water)

A: Dijkstra's，用路径中最大值作为当前距离。

```cpp
class Solution {
public:
    int swimInWater(vector<vector<int>>& grid) {
        int n = grid.size();
        if (n == 1) {
            return 0;
        }
        
        vector<vector<bool>> visited(n, vector<bool>(n));
        visited[0][0] = true;
        
        int result = max(grid[0][0], grid[n - 1][n - 1]);
        
        priority_queue<vector<int>, vector<vector<int>>, greater<vector<int>>> pq;
        pq.push({result, 0, 0});
        
        while (!pq.empty()) {
            vector<int> curr = pq.top();
            pq.pop();
            
            result = max(result, curr[0]);
            
            for (int i = 0; i < 4; i++) {
                int x = curr[1] + dirs[i][0];
                int y = curr[2] + dirs[i][1];
                
                if (x < 0 || x >= n || y < 0 || y >= n || visited[x][y]) {
                    continue;
                }
                
                if (x == n - 1 && y == n - 1) {
                    return result;
                }

                pq.push({grid[x][y], x, y});
                visited[x][y] = true;
            }
        }
        
        return -1;
    }
private:
    vector<vector<int>> dirs = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
};
```

## [Cheapest Flights Within K Stops](https://leetcode.com/problems/cheapest-flights-within-k-stops)

A: Dijkstra's，花费更少及靠站更少两种情况均存入优先队列。

```cpp
class Solution {
public:
    int findCheapestPrice(int n, vector<vector<int>>& flights, int src, int dst, int k) {
        // build adjacency matrix
        vector<vector<int>> adj(n, vector<int>(n));
        for (int i = 0; i < flights.size(); i++) {
            vector<int> flight = flights[i];
            adj[flight[0]][flight[1]] = flight[2];
        }
        
        // shortest distances
        vector<int> distances(n, INT_MAX);
        distances[src] = 0;
        // shortest steps
        vector<int> currStops(n, INT_MAX);
        currStops[src] = 0;
        
        // priority queue -> (cost, node, stops)
        priority_queue<vector<int>, vector<vector<int>>, greater<vector<int>>> pq;
        pq.push({0, src, 0});
        
        while (!pq.empty()) {
            int cost = pq.top()[0];
            int node = pq.top()[1];
            int stops = pq.top()[2];
            pq.pop();
            
            // if destination is reached, return cost to get here
            if (node == dst) {
                return cost;
            }
            
            // if no more steps left, continue
            if (stops == k + 1) {
                continue;
            }
            
            // check & relax all neighboring edges
            for (int neighbor = 0; neighbor < n; neighbor++) {
                if (adj[node][neighbor] > 0) {
                    int currCost = cost;
                    int neighborDist = distances[neighbor];
                    int neighborWeight = adj[node][neighbor];
                    
                    // check if better cost
                    int currDist = currCost + neighborWeight;
                    if (currDist < neighborDist || stops + 1 < currStops[neighbor]) {
                        pq.push({currDist, neighbor, stops + 1});
                        distances[neighbor] = currDist;
                        currStops[neighbor] = stops;
                    } else if (stops < currStops[neighbor]) {
                        // check if better steps
                        pq.push({currDist, neighbor, stops + 1});
                    }
                    currStops[neighbor] = stops;
                }
            }
        }
        
        if (distances[dst] == INT_MAX) {
            return -1;
        }
        return distances[dst];
    }
};
```
