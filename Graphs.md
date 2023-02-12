# Graphs

## [Number of Islands](https://leetcode.com/problems/number-of-islands)

A: DFS，第一次登岛时ans++。

```cpp
class Solution {
public:
    int numIslands(vector<vector<char>>& grid) {
        int ans = 0;
        int m = grid.size(), n = grid[0].size();
        for (int row = 0; row < m; row++) {
            for (int col = 0; col < n; col++) {
                dfs(grid, ans, row, col, m, n, true);
            }
        }
        return ans;
    }
private:
    void dfs(vector<vector<char>>& grid, int &ans, int i, int j, int m, int n, bool isRoot) {
        if (i < 0 || i >= m || j < 0 || j >= n || grid[i][j] == '0') {
            return;
        }

        grid[i][j] = '0';
        dfs(grid, ans, i+1, j, m, n, false);
        dfs(grid, ans, i, j+1, m, n, false);
        dfs(grid, ans, i-1, j, m, n, false);
        dfs(grid, ans, i, j-1, m, n, false);
        if (isRoot) ans++;
    }
};
```

## [Clone Graph](https://leetcode.com/problems/clone-graph)

A: DFS，哈希表存放结点状态。

```cpp
/*
// Definition for a Node.
class Node {
public:
    int val;
    vector<Node*> neighbors;
    Node() {
        val = 0;
        neighbors = vector<Node*>();
    }
    Node(int _val) {
        val = _val;
        neighbors = vector<Node*>();
    }
    Node(int _val, vector<Node*> _neighbors) {
        val = _val;
        neighbors = _neighbors;
    }
};
*/

class Solution {
public:
    Node* cloneGraph(Node* node) {
        unordered_map<int, Node*>nodes;

        return dfs(node, nodes);
    }

private:
    Node* dfs(Node* node, unordered_map<int, Node*> &nodes) {
        if (!node)  return nullptr;
        if (nodes.find(node->val) != nodes.end()) {
            return nodes[node->val];
        }

        Node* cNode = new Node(node->val);
        nodes[node->val] = cNode;

        for (auto n : node->neighbors) {
            cNode->neighbors.push_back(dfs(n, nodes));
        }

        return cNode;
    }
};
```

## [Pacific Atlantic Water Flow](https://leetcode.com/problems/pacific-atlantic-water-flow)

A: 从边界（海洋）开始反向DFS，寻找能到达单一海洋的集合，取交集。

```cpp
class Solution {
public:
    vector<vector<int>> pacificAtlantic(vector<vector<int>>& heights) {
        int m = heights.size();
        int n = heights[0].size();
        
        vector<vector<bool>> pacific(m, vector<bool>(n));
        vector<vector<bool>> atlantic(m, vector<bool>(n));
        
        for (int i = 0; i < m; i++) {
            dfs(heights, pacific, i, 0, m, n);
            dfs(heights, atlantic, i, n - 1, m, n);
        }
        
        for (int j = 0; j < n; j++) {
            dfs(heights, pacific, 0, j, m, n);
            dfs(heights, atlantic, m - 1, j, m, n);
        }
        
        vector<vector<int>> result;
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (pacific[i][j] && atlantic[i][j]) {
                    result.push_back({i, j});
                }
            }
        }
        
        return result;
    }
private:
    void dfs(vector<vector<int>>& heights, vector<vector<bool>>& visited,
        int i, int j, int m, int n) {
        
        visited[i][j] = true;
        
        if (i > 0 && !visited[i - 1][j] && heights[i - 1][j] >= heights[i][j]) {
            dfs(heights, visited, i - 1, j, m, n);
        }
        if (i < m - 1 && !visited[i + 1][j] && heights[i + 1][j] >= heights[i][j]) {
            dfs(heights, visited, i + 1, j, m, n);
        }
        if (j > 0 && !visited[i][j - 1] && heights[i][j - 1] >= heights[i][j]) {
            dfs(heights, visited, i, j - 1, m, n);
        }
        if (j < n - 1 && !visited[i][j + 1] && heights[i][j + 1] >= heights[i][j]) {
            dfs(heights, visited, i, j + 1, m, n);
        }
    }
};
```

## [Course Schedule](https://leetcode.com/problems/course-schedule)

A: DFS查找有向图中是否存在循环。

```cpp
class Solution {
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        // map each course to prereq list
        unordered_map<int, vector<int>> m;
        for (int i = 0; i < prerequisites.size(); i++) {
            m[prerequisites[i][0]].push_back(prerequisites[i][1]);
        }
        // all courses along current DFS path
        unordered_set<int> visited;
        
        for (int course = 0; course < numCourses; course++) {
            if (!dfs(course, m, visited)) {
                return false;
            }
        }
        return true;
    }
private:
    bool dfs(int course, unordered_map<int, vector<int>>& m, unordered_set<int>& visited) {
        if (visited.find(course) != visited.end()) {
            return false;
        }
        if (m[course].empty()) {
            return true;
        }
        visited.insert(course);
        for (int i = 0; i < m[course].size(); i++) {
            int nextCourse = m[course][i];
            if (!dfs(nextCourse, m, visited)) {
                return false;
            }
        }
        // 减少后续dfs开销
        m[course].clear();
        
        visited.erase(course);
        return true;
    }
};
```

## Number of Connected Components In An Undirected Graph

![Number of Connected Components In An Undirected Graph](fig/323.png)

A: 并查集。

```cpp
class Solution {
public:
    int countComponents(int n, vector<vector<int>>& edges) {
        vector<int> parents;
        vector<int> ranks;
        for (int i = 0; i < n; i++) {
            parents.push_back(i);
            ranks.push_back(1);
        }
        
        int result = n;
        for (int i = 0; i < edges.size(); i++) {
            int n1 = edges[i][0];
            int n2 = edges[i][1];
            result -= doUnion(parents, ranks, n1, n2);
        }
        return result;
    }
private:
    int doFind(vector<int>& parents, int n) {
        int p = parents[n];

        // 找到所在集合的parent
        while (p != parents[p]) {
            parents[p] = parents[parents[p]];
            p = parents[p];
        }
        return p;
    }
    
    int doUnion(vector<int>& parents, vector<int>& ranks, int n1, int n2) {
        int p1 = doFind(parents, n1);
        int p2 = doFind(parents, n2);

        // 同一集合无需合并
        if (p1 == p2) {
            return 0;
        }
        

        // 合并两个集合
        if (ranks[p1] > ranks[p2]) {
            parents[p2] = p1;
            ranks[p1] += ranks[p2];
        } else {
            parents[p1] = p2;
            ranks[p2] += ranks[p1];
        }
        
        return 1;
    }
};
```

## [Graph Valid Tree](https://www.lintcode.com/problem/178)

A: DFS，检查（1）是否全部遍历（2）是否存在环。

```cpp
class Solution {
public:
    bool validTree(int n, vector<vector<int>>& edges) {
        vector<vector<int>> adj(n);
        for (int i = 0; i < edges.size(); i++) {
            vector<int> edge = edges[i];
            adj[edge[0]].push_back(edge[1]);
            adj[edge[1]].push_back(edge[0]);
        }
        
        vector<bool> visited(n);
        if (hasCycle(adj, visited, -1, 0)) {
            return false;
        }
        
        for (int i = 0; i < visited.size(); i++) {
            if (!visited[i]) {
                return false;
            }
        }
        return true;
    }
private:
    bool hasCycle(vector<vector<int>>& adj, vector<bool>& visited, int parent, int child) {
        if (visited[child]) {
            return true;
        }
        visited[child] = true;
        // checking for cycles and connectedness
        for (int i = 0; i < adj[child].size(); i++) {
            int curr = adj[child][i];
            if (curr != parent && hasCycle(adj, visited, child, curr)) {
                return true;
            }
        }
        return false;
    }
};
```

## [All Paths From Source to Target](https://leetcode.com/problems/all-paths-from-source-to-target)

A: DFS + 回溯。

```cpp
class Solution {
public:
    void dfs(vector<vector<int>>&ans, vector<int>&path, int n, int source, vector<vector<int>> &graph) {
        if(source == n-1) {
            ans.push_back(path);
            return;
        }
        
        for(int x : graph[source]) {
            path.push_back(x);
            dfs(ans, path, n, x, graph);
            path.pop_back();
        }
    }
    vector<vector<int>> allPathsSourceTarget(vector<vector<int>>& graph) {
        vector<vector<int>> ans;
        vector<int> path;
        path.push_back(0);
        int n = graph.size();
        dfs(ans, path, n, 0, graph);
        return ans;
    }
};
```

## [Max Area of Island](https://leetcode.com/problems/max-area-of-island)

A: 递归函数直接返回当前岛屿面积。

```cpp
class Solution {
public:
    int maxAreaOfIsland(vector<vector<int>>& grid) {
        int m = grid.size();
        int n = grid[0].size();
        
        int result = 0;
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) {
                    result = max(result, dfs(grid, i, j, m, n));
                }
            }
        }
        
        return result;
    }
private:
    int dfs(vector<vector<int>>& grid, int i, int j, int m, int n) {
        if (i < 0 || i >= m || j < 0 || j >= n || grid[i][j] == 0) {
            return 0;
        }
        grid[i][j] = 0;
        
        return 1 + dfs(grid, i - 1, j, m, n) + dfs(grid, i + 1, j, m, n)
            + dfs(grid, i, j - 1, m, n) + dfs(grid, i, j + 1, m, n);
    }
};
```

## [Surrounded Regions](https://leetcode.com/problems/surrounded-regions)

A: 从边界搜索岛屿，给予特殊标记。

```cpp
class Solution {
public:
    void solve(vector<vector<char>>& board) {
        int m = board.size();
        int n = board[0].size();
        
        // marking escaped cells along the border
        for (int i = 0; i < m; i++) {
            dfs(board,i,0,m,n);
            dfs(board,i,n-1,m,n);
        }
        
        for (int j = 0; j < n; j++) {
            dfs(board,0,j,m,n);
            dfs(board,m-1,j,m,n);
        }
        
        // flip cells to correct final states
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] == 'O') {
                    board[i][j] = 'X';
                }
                if (board[i][j] == 'E') {
                    board[i][j] = 'O';
                }
            }
        }
    }
private:
    void dfs(vector<vector<char>>& board, int i, int j, int m, int n) {
        if (i < 0 || i >= m || j < 0 || j >= n || board[i][j] != 'O') {
            return;
        }
        
        board[i][j] = 'E';
        
        dfs(board, i - 1, j, m, n);
        dfs(board, i + 1, j, m, n);
        dfs(board, i, j - 1, m, n);
        dfs(board, i, j + 1, m, n);
    }
};
```

## [Rotting Oranges](https://leetcode.com/problems/rotting-oranges)

A: BFS，用队列存储腐烂橘子，记录同一时刻腐烂橘子个数。

```cpp
class Solution {
public:
    int orangesRotting(vector<vector<int>>& grid) {
        vector<int> dir = {-1, 0, 1, 0, -1}; //used for finding all 4 adjacent coordinates
        
        int m=grid.size();
        int n=grid[0].size();
        
        queue<pair<int,int>> q;
        int fresh=0; //To keep track of all fresh oranges left
        for (int i = 0;i<m;i++)
            for (int j=0;j<n;j++) {
                if(grid[i][j]==2)
                    q.push({i,j});
                if(grid[i][j]==1)
                    fresh++;
            }
        int ans=-1; //initialised to -1 since after each step we increment the time by 1 and initially all rotten oranges started at 0.
        while(!q.empty()) {
            int sz=q.size(); // 同一时刻腐烂橘子个数，全部出队列后时间流逝
            while(sz--) {
                pair<int,int> p=q.front();
                q.pop();
                for(int i=0;i<4;i++)
                {
                    // BFS，腐烂邻近橘子
                    int r=p.first+dir[i];
                    int c=p.second+dir[i+1];
                    if(r>=0 && r<m && c>=0 && c<n && grid[r][c]==1)
                    {
                        grid[r][c]=2;
                        q.push({r,c});
                        fresh--; // decrement by 1 foreach fresh orange that now is rotten
                    }
                    
                }
            }
            ans++; //incremented after each minute passes
        }
        if(fresh>0) return -1; //if fresh>0 that means there are fresh oranges left
        if(ans==-1) return 0; //we initialised with -1, so if there were no oranges it'd take 0 mins.
        return ans;
        
    }
};
```

## [Walls And Gates](https://www.lintcode.com/problem/663)

A: 从门开始执行BFS，更新最短距离。

```cpp
class Solution {
public:
    const int INF = 2147483647;
    vector<int> dir = {-1, 0, 1, 0, -1};
    void wallsAndGates(vector<vector<int>> &rooms) {
        int m = rooms.size(), n = rooms[0].size();
        std::queue<pair<int, int>> dq;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (rooms[i][j] == 0) {
                    dq.push({i, j});
                }
            }
        }
        while (!dq.empty()) {
            int len = dq.size();
            while (len--) {
                pair<int, int> room = dq.front();
                dq.pop();
                for (int i = 0; i < 4; i++) {
                    int row = room.first + dir[i];
                    int col = room.second + dir[i + 1];
                    if (row >= 0 && row < m && col >= 0 && col < n) {
                        if (rooms[row][col] == -1 || rooms[row][col] == 0) continue;
                        else {
                            if (rooms[row][col] > rooms[room.first][room.second] + 1) {
                                dq.push({row, col});
                                rooms[row][col] = rooms[room.first][room.second] + 1;
                            }
                        }
                    }
                }
            }
        }
    }
};
```

## [Course Schedule II](https://leetcode.com/problems/course-schedule-ii)

A: **拓扑排序**，通过dfs寻找叶子结点，回溯加入visited(前置课程实际上为子节点)。

```cpp
class Solution {
public:
    vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {
        unordered_map<int, vector<int>> m;
        // build adjacency list of prereqs
        for (int i = 0; i < prerequisites.size(); i++) {
            m[prerequisites[i][0]].push_back(prerequisites[i][1]);
        }
        unordered_set<int> visit;
        unordered_set<int> cycle;
        
        vector<int> result;
        for (int course = 0; course < numCourses; course++) {
            if (!dfs(course, m, visit, cycle, result)) {
                return {};
            }
        }
        return result;
    }
private:
    // a course has 3 possible states:
    // visited -> course added to result
    // visiting -> course not added to result, but added to cycle
    // unvisited -> course not added to result or cycle
    bool dfs(int course, unordered_map<int, vector<int>>& m, unordered_set<int>& visit,
        unordered_set<int>& cycle, vector<int>& result) {
        
        if (cycle.find(course) != cycle.end()) {
            return false;
        }
        if (visit.find(course) != visit.end()) {
            return true;
        }
        cycle.insert(course);
        for (int i = 0; i < m[course].size(); i++) {
            int nextCourse = m[course][i];
            if (!dfs(nextCourse, m, visit, cycle, result)) {
                return false;
            }
        }
        cycle.erase(course);
        visit.insert(course);
        result.push_back(course);
        return true;
    }
};
```

## [Redundant Connection](https://leetcode.com/problems/redundant-connection)

A: 并查集找最小生成树。

```cpp
class Solution {
public:
    vector<int> findRedundantConnection(vector<vector<int>>& edges) {
        int n = edges.size();
        
        vector<int> parents;
        vector<int> ranks;
        for (int i = 0; i < n + 1; i++) {
            parents.push_back(i);
            ranks.push_back(1);
            
        }
        
        vector<int> result;
        for (int i = 0; i < n; i++) {
            int n1 = edges[i][0];
            int n2 = edges[i][1];
            if (!doUnion(parents, ranks, n1, n2)) {
                result = {n1, n2};
                break;
            }
        }
        return result;
    }
private:
    int doFind(vector<int>& parents, int n) {
        int p = parents[n];
        while (p != parents[p]) {
            parents[p] = parents[parents[p]];
            p = parents[p];
        }
        return p;
    }
    
    bool doUnion(vector<int>& parents, vector<int>& ranks, int n1, int n2) {
        int p1 = doFind(parents, n1);
        int p2 = doFind(parents, n2);
        if (p1 == p2) {
            return false;
        }
        
        if (ranks[p1] > ranks[p2]) {
            parents[p2] = p1;
            ranks[p1] += ranks[p2];
        } else {
            parents[p1] = p2;
            ranks[p2] += ranks[p1];
        }
        
        return true;
    }
};
```

## [Word Ladder](https://leetcode.com/problems/word-ladder)

A: BFS寻找最短路径。

```cpp
class Solution {
public:
    int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
        unordered_set<string> dict;
        for (int i = 0; i < wordList.size(); i++) {
            dict.insert(wordList[i]);
        }
        
        queue<string> q;
        q.push(beginWord);
        
        int result = 1;
        
        while (!q.empty()) {
            int count = q.size();
            
            for (int i = 0; i < count; i++) {
                string word = q.front();
                q.pop();
                
                if (word == endWord) {
                    return result;
                }
                dict.erase(word);
                
                for (int j = 0; j < word.size(); j++) {
                    char c = word[j];
                    for (int k = 0; k < 26; k++) {
                        word[j] = k + 'a';
                        if (dict.find(word) != dict.end()) {
                            q.push(word);
                            dict.erase(word);
                        }
                        word[j] = c;
                    }
                }
            }
            
            result++;
        }
        
        return 0;
    }
};
```

## [Count Sub Islands](https://leetcode.com/problems/count-sub-islands)

A: DFS，用与运算标定两张格网中同时为1的元素。

```cpp
class Solution {
public:
        int countSubIslands(vector<vector<int>>& B, vector<vector<int>>& A) {
        int m = A.size(), n = A[0].size(), res = 0;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                if (A[i][j] == 1)
                    res += dfs(B, A, i, j);
        return res;
    }

    int dfs(vector<vector<int>>& B, vector<vector<int>>& A, int i, int j) {
        int m = A.size(), n = A[0].size(), res = 1;
        if (i < 0 || i == m || j < 0 || j == n || A[i][j] == 0) return 1;
        A[i][j] = 0;
        res &= dfs(B, A, i - 1, j);
        res &= dfs(B, A, i + 1, j);
        res &= dfs(B, A, i, j - 1);
        res &= dfs(B, A, i, j + 1);
        return res & B[i][j];
    }
};
```

## [Reorder Routes to Make All Paths Lead to The City Zero](https://leetcode.com/problems/reorder-routes-to-make-all-paths-lead-to-the-city-zero)

A: 将边视作双向，dfs，实际上无法到达则ans++。

```cpp
class Solution {
public:
    int dfs(vector<vector<int>> &al, vector<bool> &visited, int from) {
        auto change = 0;
        visited[from] = true;
        for (auto to : al[from])
            if (!visited[abs(to)])
                change += dfs(al, visited, abs(to)) + (to > 0);
        return change;        
    }
    int minReorder(int n, vector<vector<int>>& connections) {
        vector<vector<int>> al(n);
        for (auto &c : connections) {
            al[c[0]].push_back(c[1]);
            al[c[1]].push_back(-c[0]); // 负号表示实际不存在
        }
        return dfs(al, vector<bool>(n) = {}, 0);
    }
};
```

## [Snakes And Ladders](https://leetcode.com/problems/snakes-and-ladders)

A: BFS。

```cpp
class Solution {
public:
    int snakesAndLadders(vector<vector<int>>& board) {
        int n=board.size();
        vector<int> vis((n*n)+1,0);
        int ans=0;
        queue<int> q;
        q.push(1);
        vis[1]=1;
        while(!q.empty())
        {
            ans++;
            int t=q.size();
            while(t--)
            {
                int curr=q.front();
                q.pop();
                for(int i=curr+1; i<=curr+6  ;i++)
                {
                        if(i==n*n) return ans;
                        int x=n-ceil(1.0*i/n*1.0);
                        int y=0;
                        if(n%2==0)
                            y=(x%2!=0?(i-1)%n : n-(i-1)%n-1);
                        else
                            y=(x%2!=0)?( n-(i-1)%n-1 ):(i-1)%n;
                    
                        int dest= board[x][y]!=-1?board[x][y]:i;
                    
                        if(dest==n*n) return ans;
                        if(vis[dest]==0)
                        {
                           q.push(dest);
                            vis[dest]=1;
                        }                      
                    
                }
                
            }
         
        }
         return -1;
    }
};
```

## [Open The Lock](https://leetcode.com/problems/open-the-lock)

A: BFS。

```cpp
class Solution {
public:
    int openLock(vector<string>& deadends, string target) {
        unordered_set<string> s;
        for (auto &d : deadends) {
            s.insert(d);
        }
        if (s.find("0000") != s.end()) return -1;
        if (target == "0000") return 0;
        queue<string> q;
        q.push("0000");
        int ans = 0;
        while (!q.empty()) {
            ans++;
            int len = q.size();
            while (len--) {
                string cur = q.front();
                q.pop();
                for (int i = 0; i < 4; i++) {
                    string in = increase(cur, i);
                    string de = decrease(cur, i);
                    if (in == target || de == target) {
                        return ans;
                    }
                    if (s.find(in) == s.end()) {
                        s.insert(in);
                        q.push(in);
                    }
                    if (s.find(de) == s.end()) {
                        s.insert(de);
                        q.push(de);
                    }
                }
            }
        }
        return -1;
    }
private:
    string increase(string cur, int i) {
        string ans = cur;
        if (ans[i] == '9') {
            ans[i] = '0';
        } else {
            ans[i]++;
        }
        return ans;
    }
    
    string decrease(string cur, int i) {
        string ans = cur;
        if (ans[i] == '0') {
            ans[i] = '9';
        } else {
            ans[i]--;
        }
        return ans;
    }
};
```

## [Find Eventual Safe States](https://leetcode.com/problems/find-eventual-safe-states)

A: 不在环中的结点为目标节点。

```cpp
class Solution {
public:
    using vvi = vector<vector<int>>;
    using usi = unordered_set<int>;

    usi cycle_nodes;
    usi safe_nodes;

    bool dfs (const vvi& g, int i, usi visited_nodes) 
    {    
        if (safe_nodes.find (i)  != safe_nodes.end ())  return true;  // we know safe already
        if (cycle_nodes.find (i) != cycle_nodes.end ()) return false; // we know in cycle already

        if (visited_nodes.find (i) != visited_nodes.end ()) {         // we have determined node is in cycle
            cycle_nodes.insert (i);
            return false;
        }

        visited_nodes.insert (i); // keep track of nodes we've visited already

        for (int node : g[i]) {
            if (!dfs (g, node, visited_nodes)) {
                cycle_nodes.insert (i); // if child is in cycle, parent must be too
                return false;
            }
        }

        safe_nodes.insert (i); // we know node is safe now

        return true;
    }

    vector<int> eventualSafeNodes(vvi& graph) 
    {
        vector<int> ans;
        usi visited_nodes;

        for (int i = 0; i < graph.size (); i++) {
            if (dfs (graph, i, visited_nodes)) ans.push_back (i);
        }

        return ans;
    }
};
```

## [Find the Town Judge](https://leetcode.com/problems/find-the-town-judge)

A: 统计入度出度，差值为n-1的为目标。

```cpp
int findJudge(int N, vector<vector<int>>& trust) {
    vector<int> count(N + 1, 0);
    for (auto& t : trust)
        count[t[0]]--, count[t[1]]++;
    for (int i = 1; i <= N; ++i) {
        if (count[i] == N - 1) return i;
    }
    return -1;
}
```

## [Find Closest Node to Given Two Nodes](https://leetcode.com/problems/find-closest-node-to-given-two-nodes)

A: BFS，分别记录两个节点历史路径以确定答案。

```cpp
class Solution {
public:
    int closestMeetingNode(vector<int>& edges, int node1, int node2) {
        vector<bool> dist1(edges.size(), 0);
        vector<bool> dist2(edges.size(), 0);
        queue<int> q1;
        queue<int> q2;
        q1.push(node1);
        q2.push(node2);
        while (!q1.empty() || !q2.empty()) {
            int res1 = bfs(q1, edges, dist1, dist2);
            int res2 = bfs(q2, edges, dist2, dist1);
            cout << res1 << ", " << res2 << endl;
            if (res1 != -1 && res2 != -1) return min(res1, res2);
            if (res1 != -1) return res1;
            if (res2 != -1) return res2;
        }
        return -1;
    }
private:
    int bfs(queue<int> &q, vector<int> &edges, vector<bool> &dist1, vector<bool> &dist2) {
        if (q.empty()) return -1;
        int n = q.front();
        q.pop();
        if (dist2[n] == 1) {
            return n;
        } else {
            dist1[n] = 1;
            if (edges[n] != -1 && dist1[edges[n]] == 0) {
                q.push(edges[n]);
            }
            return -1;
        }
    }
};
```

## [Check if Move is Legal](https://leetcode.com/problems/check-if-move-is-legal)

A: 分八个方向依次按条件进行DFS。

```cpp
class Solution {
public:
    bool checkMove(vector<vector<char>>& board, int rMove, int cMove, char color) {
        vector<pair<int, int>> dirs = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
        for (auto &dir : dirs) {
            int r = rMove + dir.first, c = cMove + dir.second;
            if (r < 0 || r == 8 || c < 0 || c == 8 || board[r][c] == '.' || board[r][c] == color) continue;
            if (dfs(board, r + dir.first, c + dir.second, color, dir)) return true;
        }
        return false;
    }
private:
    bool dfs(vector<vector<char>> &board, int r, int c, char color, pair<int, int> &dir) {
        if (r < 0 || r == 8 || c < 0 || c == 8 || board[r][c] == '.') {
            return false;
        }
        if (board[r][c] == color) {
            return true;
        }
        return dfs(board, r+dir.first, c+dir.second, color, dir);
    }
};
```

## [Shortest Bridge](https://leetcode.com/problems/shortest-bridge)

A: 首先DFS找到一个岛屿，以该岛屿边界为起点用BFS寻找下一个岛屿。即可得到最短路径。

```cpp
class Solution {
    vector<vector<int> > mat;
    vector<vector<int> > vis;
    int m,n;
    int x[4]={-1,0,1,0};
    int y[4]={0,1,0,-1};
    queue<pair<int,int> > que;
public:
    void dfs(int i,int j){
        vis[i][j]=1;
        que.push({i,j});
        for(int dir=0;dir<4;dir++){
            int xd=i+x[dir];
            int yd=j+y[dir];
            if(xd>=0 && yd>=0 && xd<=m-1 && yd<=n-1){
                if(!vis[xd][yd] && mat[xd][yd]==1){
                    dfs(xd,yd);
                }
            }
        }
    }
    int shortestBridge(vector<vector<int>>& A) {
        m=A.size();
        if(m==0) return 0;
        n=A[0].size();
        cout<<m<<" "<<n;
        mat=A;
        vis.resize(m,vector<int>(n,0));
        int flag=0;
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                if(mat[i][j]==1){
                    dfs(i,j);
                    flag=1;
                    break;
                }
            }
            if(flag){
                break;
            }
        }
        int l=0;
        while(!que.empty()){
            int len=que.size();
            l++;
            while(len--){
                pair<int,int> poi=que.front();
                que.pop();
                vis[poi.first][poi.second]=1;
                for(int dir=0;dir<4;dir++){
                    int xd=poi.first+x[dir];
                    int yd=poi.second+y[dir];
                    if(xd>=0 && yd>=0 && xd<=m-1 && yd<=n-1){
                       if(!vis[xd][yd] && mat[xd][yd]==1){
                           return l-1;
                       }
                       else if(!vis[xd][yd] && mat[xd][yd]==0){
                           vis[xd][yd]=1;
                           que.push({xd,yd});
                       }
                    }
                }
            }
        }
        return -1;
    }
};
```

## [Minimum Number of Days to Eat N Oranges](https://leetcode.com/problems/minimum-number-of-days-to-eat-n-oranges)

A: DP。

```cpp
class Solution {
public:
    unordered_map<int, int> dp;
    int minDays(int n) {
        if (n <= 1)
            return n;
        if (dp.count(n) == 0)
            dp[n] = 1 + min(n % 2 + minDays(n / 2), n % 3 + minDays(n / 3));
        return dp[n];
    }
};
```

## [Island Perimeter](https://leetcode.com/problems/island-perimeter)

A: DFS。

```cpp
class Solution {
public:
    int islandPerimeter(vector<vector<int>>& grid) {
        int ans = 0;
        int m = grid.size(), n = grid[0].size();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1)
                    dfs(ans, grid, m, n, i, j);
            }
        }
        return ans;
    }
private:
    void dfs(int &ans, vector<vector<int>>& grid, int m, int n, int i, int j) {
        if (i < 0 || i == m || j < 0 || j == n || grid[i][j] == 0) {
            ans++;
            return;
        }
        if (grid[i][j] == -1) return;
        grid[i][j] = -1;
        dfs(ans, grid, m, n, i + 1, j);
        dfs(ans, grid, m, n, i - 1, j);
        dfs(ans, grid, m, n, i, j + 1);
        dfs(ans, grid, m, n, i, j - 1);
    }
};
```

## [Verifying an Alien Dictionary](https://leetcode.com/problems/verifying-an-alien-dictionary)

A: 将新的顺序还原成我们常用的字母顺序，然后判断是否排序。

```cpp
class Solution {
public:
    bool isAlienSorted(vector<string> words, string order) {
        int mapping[26];
        for (int i = 0; i < 26; i++)
            mapping[order[i] - 'a'] = i;
        for (string &w : words)
            for (char &c : w)
                c = mapping[c - 'a'];
        return is_sorted(words.begin(), words.end());
    }
};
```

## [Shortest Path in Binary Matrix](https://leetcode.com/problems/shortest-path-in-binary-matrix)

A: BFS，计算最短路径问题。

```cpp
class Solution {
public:
    int shortestPathBinaryMatrix(vector<vector<int>>& g, int steps = 0) {
      queue<pair<int, int>> q;
      q.push({ 0, 0 });
      while (!q.empty()) {
        ++steps;
        queue<pair<int, int>> q1;
        while (!q.empty()) {
          auto c = q.front();
          q.pop();
          if (exchange(g[c.first][c.second], 1) == 1) continue; // exchange: Replaces the value of obj with new_value and returns the old value of obj.
          if (c.first == g.size() - 1 && c.second == g.size() - 1) return steps;
          for (auto i = c.first - 1; i <= c.first + 1; ++i)
            for (auto j = c.second - 1; j <= c.second + 1; ++j)
              if (i != c.first || j != c.second) {
                if (i >= 0 && j >= 0 && i < g.size() && j < g.size() && !g[i][j]) {
                  q1.push({ i, j });
                }
              }
        }
        swap(q, q1);
      }
      return -1;
    }
};
```

## [As Far from Land as Possible](https://leetcode.com/problems/as-far-from-land-as-possible)

A: BFS，将所有陆地入队，然后从陆地开始向四周扩散，直到遇到海洋。直到最远的陆地都扩散完，就是最远距离。

```go
func maxDistance(grid [][]int) int {
    var m int = len(grid)
    var n int = len(grid[0])
    Queue := [][]int{}
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            if grid[i][j] == 1 {
                Queue = append(Queue, []int{i, j})
            }
        }
    }
    if len(Queue) == m * n {return -1}
    var ans int = -1
    dirs := [][]int{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}
    for len(Queue) > 0 {
        ans++
        var s int = len(Queue)
        for i := 0; i < s; i++ {
            curr := Queue[0]
            Queue = Queue[1: len(Queue)]
            for _, dir := range dirs {
                var x int = curr[0] + dir[0]
                var y int = curr[1] + dir[1]
                if x >= 0 && y >= 0 && x < m && y < n && grid[x][y] == 0 {
                    grid[x][y] = 1
                    Queue = append(Queue, []int{x, y})
                }
            }
        }
    }
    return ans
}
```

## [Shortest Path with Alternating Colors](https://leetcode.com/problems/shortest-path-with-alternating-colors)

A: BFS，v1、v2分别存储两种颜色下各个结点的访问情况。

```cpp

class Solution {
public:
    vector<int> shortestAlternatingPaths(int n, vector<vector<int>>& red_edges, vector<vector<int>>& blue_edges){

        vector<int> res(n, -1), v1(n, 0), v2(n, 0);
        vector<vector<pair<int, int>>> vp(n); // pair<to, color>
        queue<pair<int, int>> q;

        for(auto &it : red_edges) vp[it[0]].push_back({it[1], 1});
        for(auto &it : blue_edges) vp[it[0]].push_back({it[1], 2});

        res[0] = 0; // 初始位置
        v1[0] = 1;
        v2[0] = 1;
        int temp = 1;

        for(auto &it : vp[0])
        {
            q.push(it);
            if(res[it.first] == -1) res[it.first] = temp;
        }

        while(!q.empty())
        {
            int len = q.size();
            temp++;
            for(int i=0; i<len; ++i)
            {
                auto val = q.front();
                q.pop();
                if(val.second == 1) v1[val.first] = 1;
                else v2[val.first] = 1;

                for(auto &it : vp[val.first])
                {
                    // 跳过相同颜色路径
                    if(val.second == 1 && (v2[it.first] || it.second==1)) continue;
                    if(val.second == 2 && (v1[it.first] || it.second==2)) continue;
                    q.push(it);                      
                    if(res[it.first] == -1) res[it.first] = temp;
                }
            }
        }
        return res;
    }
};
```

## [Minimum Fuel Cost to Report to the Capital](https://leetcode.com/problems/minimum-fuel-cost-to-report-to-the-capital)

A: 

```cpp
class Solution {
 public:
  long long minimumFuelCost(vector<vector<int>>& roads, int seats) {
    long long ans = 0;
    vector<vector<int>> graph(roads.size() + 1);

    for (const vector<int>& road : roads) {
      const int u = road[0];
      const int v = road[1];
      graph[u].push_back(v);
      graph[v].push_back(u);
    }

    dfs(graph, 0, -1, seats, ans);
    return ans;
  }

 private:
  int dfs(const vector<vector<int>>& graph, int u, int prev, int seats,
          long long& ans) {
    int people = 1;
    for (const int v : graph[u]) {
      if (v == prev)
        continue;
      people += dfs(graph, v, u, seats, ans);
    }
    if (u > 0)
      // # of cars needed = ceil(people / seats)
      ans += (people + seats - 1) / seats;
    return people;
  }
};
```

```go
func minimumFuelCost(roads [][]int, seats int) int64 {
    var ans int64 = 0
    adj := make(map[int][]int)
    // initialize the adjacency list
    for _, v := range roads {
        if _, ok := adj[v[0]]; !ok {
            adj[v[0]] = make([]int, 0)
        }
        if _, ok := adj[v[1]]; !ok {
            adj[v[1]] = make([]int, 0)
        }
        adj[v[0]] = append(adj[v[0]], v[1])
        adj[v[1]] = append(adj[v[1]], v[0])
    }
    dfs(int64(seats), adj, &ans, 0, -1)
    return ans
}

// minimum cost at node cur
func dfs(seats int64, adj map[int][]int, ans *int64, cur, pre int) int64 {
    var load int64 = 1 // how much representatives we have at the cur node
    for _, node := range adj[cur] {
        if node == pre {
            // we only traverse children of the cur node
            continue
        }
        load += dfs(seats, adj, ans, node, cur)
    }
    if cur != 0 {
        // if cur is not the root, we need to add the cost of the car
        // ceil(load / seats)
        *ans += (load + seats - 1) / seats
    }
    return load
}
```
