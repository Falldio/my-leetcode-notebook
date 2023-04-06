# Stack

## [Valid Parentheses](https://leetcode.com/problems/valid-parentheses)

A: 左括号入栈，右括号出栈。

```cpp
class Solution {
public:
    bool isValid(string s) {
        deque<char> stack;
        int len = s.size();
        for (int i = 0; i < len; i++) {
            if (s[i] == '(' || s[i] == '[' || s[i] == '{') {
                stack.push_back(s[i]);
            }
            else {
                switch(s[i]) {
                    case ')':
                        if (!stack.empty() && stack.back() == '(')    stack.pop_back();
                        else    return false;
                        break;
                    case ']':
                        if (!stack.empty() && stack.back() == '[')    stack.pop_back();
                        else    return false;
                        break;
                    case '}':
                        if (!stack.empty() && stack.back() == '{')    stack.pop_back();
                        else    return false;
                        break;
                    default:
                        return false;
                }
            }
        }
        if (stack.empty())  return true;
        else return false;
    }
};
```

```go
func isValid(s string) bool {
    stk := []rune{}
    for _, r := range s {
        switch r {
        case '(', '[', '{':
            stk = append(stk, r)
        case ')':
            if len(stk) == 0 || stk[len(stk) - 1] != '(' {
                return false
            } else {
                stk = stk[:len(stk) - 1]
            }
        case ']':
            if len(stk) == 0 || stk[len(stk) - 1] != '[' {
                return false
            } else {
                stk = stk[:len(stk) - 1]
            }
        case '}':
            if len(stk) == 0 ||stk[len(stk) - 1] != '{' {
                return false
            } else {
                stk = stk[:len(stk) - 1]
            }
        }
    }
    return len(stk) == 0
}
```

## [Min Stack](https://leetcode.com/problems/min-stack)

A: 用另一个栈存储每次push操作的最小值。

```cpp
class MinStack {
public:
    stack<int> stack;
    std::stack<int> minStack;
    MinStack() {
        
    }
    
    void push(int val) {
        stack.push(val);
        if (minStack.empty()) {
            minStack.push(val);
        } else {
            minStack.push(min(val, minStack.top()));
        }
    }
    
    void pop() {
        stack.pop();
        minStack.pop();
    }
    
    int top() {
        return stack.top();
    }
    
    int getMin() {
        return minStack.top();
    }
};

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack* obj = new MinStack();
 * obj->push(val);
 * obj->pop();
 * int param_3 = obj->top();
 * int param_4 = obj->getMin();
 */
 ```

## [Evaluate Reverse Polish Notation](https://leetcode.com/problems/evaluate-reverse-polish-notation)

A: 遇到数字则入栈，遇到运算符则两次出栈运算，注意cpp判断字符串是否为数字的方法。

```cpp
int evalRPN(vector<string>& tokens) {
    stack<int> stn;
    for(auto s:tokens) {
        if(s.size()>1 || isdigit(s[0])) stn.push(stoi(s));
        else {
            auto x2=stn.top(); stn.pop();
            auto x1=stn.top(); stn.pop();
            switch(s[0]) {
                case '+': x1+=x2; break;
                case '-': x1-=x2; break;
                case '*': x1*=x2; break;
                case '/': x1/=x2; break;
            }
            stn.push(x1);
        }
    }
    return stn.top();
}
```

```go
func evalRPN(tokens []string) int {
    stk := []int{}
    for _, tok := range tokens {
        switch tok {
        case "+":
            n1, n2 := stk[len(stk) - 2], stk[len(stk) - 1]    
            stk = stk[:len(stk) - 2]
            stk = append(stk, n1+n2)
        case "-":
            n1, n2 := stk[len(stk) - 2], stk[len(stk) - 1]    
            stk = stk[:len(stk) - 2]
            stk = append(stk, n1-n2)
        case "*":
            n1, n2 := stk[len(stk) - 2], stk[len(stk) - 1]    
            stk = stk[:len(stk) - 2]
            stk = append(stk, n1*n2)
        case "/":
            n1, n2 := stk[len(stk) - 2], stk[len(stk) - 1]    
            stk = stk[:len(stk) - 2]
            stk = append(stk, n1/n2)
        default:
            n, _ := strconv.Atoi(tok)
            stk = append(stk, n)
        }
    }
    return stk[0]
}
```

## [Daily Temperatures](https://leetcode.com/problems/daily-temperatures)

A: Next Greater Element，从后往前遍历，用栈保存历史较大元素（通过每次遍历判断较大元素）。

[详解NGE](https://leetcode.com/problems/daily-temperatures/solutions/1574806/c-easy-standard-sol-intuitive-approach-with-dry-run-stack-appraoch/?orderBy=most_votes)

```cpp
class Solution {
public:
    vector<int> dailyTemperatures(vector<int>& temperatures) {
        
        int n = temperatures.size();
        vector<int>nge(n, 0); // initially all 0, stores distance between their next greater element and current temperature
        stack<int>st{};
        
        // move from right to left
        for(int i = n-1; i>=0; --i){
            // pop until we find next greater element to the right
            // since we came from right stack will have element from right only
            // s.top() is the index of elements so we put that index inside temperatures vector to check
            while(!st.empty() && temperatures[st.top()] <= temperatures[i])
                st.pop();

            // if stack not empty, then we have some next greater element, 
            // so we take distance between next greater and current temperature
            // as we are storing indexes in the stack
            if(!st.empty())
                nge[i] = st.top()-i; // distance between next greater and current
            
            // push the index of current temperature in the stack,
            // same as pushing current temperature in stack
            st.push(i);
        }
        
        return nge;
    }
};
```

```go
type pair struct {
    t int
    idx int
}

func dailyTemperatures(temperatures []int) []int {
    stk := []pair{}
    ans := make([]int, len(temperatures))
    for i, t := range temperatures {
        if len(stk) == 0 {
            stk = append(stk, pair{t, i})
        } else {
            for len(stk) > 0 && stk[len(stk) - 1].t < t {
                ans[stk[len(stk) - 1].idx] = i - stk[len(stk) - 1].idx
                stk = stk[:len(stk) - 1]
            }
            stk = append(stk, pair{t, i})
        }
    }
    return ans
}
```

## [Car Fleet](https://leetcode.com/problems/car-fleet)

A: 按照位置反向遍历，通过到达目的地时间合并车辆。

```cpp
class Solution {
public:
    int carFleet(int target, vector<int>& position, vector<int>& speed) {
        int n = position.size();
        
        vector<pair<int, double>> cars;
        for (int i = 0; i < n; i++) {
            double time = (double) (target - position[i]) / speed[i];
            cars.push_back({position[i], time});
        }
        sort(cars.begin(), cars.end());
        
        double maxTime = 0.0;
        int result = 0;
        
        for (int i = n - 1; i >= 0; i--) {
            double time = cars[i].second;
            if (time > maxTime) {
                maxTime = time;
                result++;
            }
        }
        
        return result;
    }
};
```

## [Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram)

A: 条形图升序时，入栈；降序时开始比较计算结果。

```cpp
class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        // pair: [index, height]
        stack<pair<int, int>> stk;
        int result = 0;
        
        for (int i = 0; i < heights.size(); i++) {
            int start = i;
            
            while (!stk.empty() && stk.top().second > heights[i]) {
                int index = stk.top().first;
                int width = i - index;
                int height = stk.top().second;
                stk.pop();
                
                result = max(result, height * width);
                start = index;
            }
            
            stk.push({start, heights[i]});
        }
        
        while (!stk.empty()) {
            int width = heights.size() - stk.top().first;
            int height = stk.top().second;
            stk.pop();
            
            result = max(result, height * width);
        }
                          
        return result;
    }
};
```

```go
func largestRectangleArea(heights []int) int {
 // 声明max并初始化为0
 max := 0
 // 使用切片实现栈
 stack := make([]int, 0)
 // 数组头部加入0
 heights = append([]int{0}, heights...)
 // 数组尾部加入0
 heights = append(heights, 0)
 // 初始化栈，序号从0开始
 stack = append(stack, 0)
 for i := 1; i < len(heights); i++ {
  // 结束循环条件为：当即将入栈元素>top元素，也就是形成非单调递增的趋势
  for heights[stack[len(stack)-1]] > heights[i] {
   // mid 是top
   mid := stack[len(stack)-1]
   // 出栈
   stack = stack[0 : len(stack)-1]
   // left是top的下一位元素，i是将要入栈的元素
   left := stack[len(stack)-1]
   // 高度x宽度
   tmp := heights[mid] * (i - left - 1)
   if tmp > max {
    max = tmp
   }
  }
  stack = append(stack, i)
 }
 return max
}
```

## [Baseball Game](https://leetcode.com/problems/baseball-game)

A: 注意+入栈顺序。

```cpp
class Solution {
public:
    int calPoints(vector<string>& ops) {
        stack<int> stack;
        int sum =  0;
        
        for (int i = 0; i < ops.size(); i++){
            if (ops[i] == "+"){
                int first = stack.top();
                stack.pop();
                
                int second = stack.top();
                
                stack.push(first);
                
                stack.push(first + second);
                
                sum += first + second;
            }
            
            else if (ops[i] == "D"){
                sum += 2 * stack.top();
                stack.push(2 * stack.top());
            }
            
            else if (ops[i] == "C"){
                sum -= stack.top();
                stack.pop();
            }
            
            else{
                sum += stoi(ops[i]);
                stack.push(stoi(ops[i]));
            }
        }
        
        return sum;
        
        
    }
};
```

## [Asteroid Collision](https://leetcode.com/problems/asteroid-collision)

A: 用vector模拟栈。

```cpp
class Solution {
public:
    vector<int> asteroidCollision(vector<int>& a) {
        vector<int> s; // use vector to simulate stack.
        for (int i = 0; i < a.size(); i++) {
            if (a[i] > 0 || s.empty() || s.back() < 0) // a[i] is positive star or a[i] is negative star and there is no positive on stack
                s.push_back(a[i]);
            else if (s.back() <= -a[i]) { // a[i] is negative star and stack top is positive star
                if(s.back() < -a[i]) i--; // only positive star on stack top get destroyed, stay on i to check more on stack.
                s.pop_back(); // destroy positive star on the frontier;
            } // else : positive on stack bigger, negative star destroyed.
        }
        return s;
    }
};
```

## [Online Stock Span](https://leetcode.com/problems/online-stock-span)

A: 栈保存历史日期和当天结果，类似于回溯。

```cpp
class StockSpanner {
public:
    StockSpanner() {
    }
    
    stack<pair<int, int>> s;
    int next(int price) {
        int res = 1;
        while (!s.empty() && s.top().first <= price) {
            res += s.top().second;
            s.pop();
        }
        s.push({price, res});
        return res;
    }
};

/**
 * Your StockSpanner object will be instantiated and called as such:
 * StockSpanner* obj = new StockSpanner();
 * int param_1 = obj->next(price);
 */
```

```go
type StockSpanner struct {
    monoStack [][2]int    
}

func Constructor() StockSpanner {
    return StockSpanner{[][2]int{}}
}

func (this *StockSpanner) Next(price int) int {
    res := 1
    for l := len(this.monoStack)-1; l > -1 && this.monoStack[l][0] <= price; l-- {
        res += this.monoStack[l][1]
        this.monoStack = this.monoStack[:l]
    }
    this.monoStack = append(this.monoStack, [2]int{price, res})
    return res   
}

// Runtime 160 ms Beats 93.51%
// Memory 8.4 MB Beats 88.31%

/**
 * Your StockSpanner object will be instantiated and called as such:
 * obj := Constructor();
 * param_1 := obj.Next(price);
 */
```

## [Simplify Path](https://leetcode.com/problems/simplify-path)

A: 用栈保存历史文件目录。

```cpp
class Solution {
public:
    string simplifyPath(string path) {
        
        stack<string> st;
        string res;
        
        for(int i = 0;  i<path.size(); ++i)
        {
            if(path[i] == '/')    
                continue;
            string temp;
            // iterate till we doesn't traverse the whole string and doesn't encounter the last /
            while(i < path.size() && path[i] != '/')
            {
            // add path to temp string
                temp += path[i];
                ++i;
            }
            if(temp == ".")
                continue;
            // pop the top element from stack if exists
            else if(temp == "..")
            {
                if(!st.empty())
                    st.pop();
            }
            else
            // push the directory file name to stack
                st.push(temp);
        }
        
        // adding all the stack elements to res
        while(!st.empty())
        {
            res = "/" + st.top() + res;
            st.pop();
        }
        
        // if no directory or file is present
        if(res.size() == 0)
            return "/";
        
        return res;
    }
};
```

## [Decode String](https://leetcode.com/problems/decode-string)

A: 以递归形式返回后续生成字符串，重复指定次数。

```cpp
class Solution {
public:
    string decodeString(const string& s, int& i) {
        string res;
        
        while (i < s.length() && s[i] != ']') {
            if (!isdigit(s[i]))
                res += s[i++];
            else {
                int n = 0;
                while (i < s.length() && isdigit(s[i]))
                    n = n * 10 + s[i++] - '0';
                    
                i++; // '['
                string t = decodeString(s, i);
                i++; // ']'
                
                while (n-- > 0)
                    res += t;
            }
        }
        
        return res;
    }

    string decodeString(string s) {
        int i = 0;
        return decodeString(s, i);
    }
};
```

## [Remove K Digits](https://leetcode.com/problems/remove-k-digits)

A: 升序栈，仅当字符串出现降序时需要移除降序部分中较大的数字。

```cpp
class Solution {
public:
    string removeKdigits(string num, int k) {
      int n = num.size();
      
      stack<char>s;
      int count = k;
      
      for(int i = 0 ; i < n; i++)
      {
        while(!s.empty() && count > 0 && s.top() > num[i])
        {
            // 在k范围内移除较大数字
          s.pop();
          count--;
        }
        s.push(num[i]);
      }
      
      // In case the num was already in a non increasing order (e.x: 123456)
      while(s.size() != n - k) s.pop();
     
      string res = "";
      while(!s.empty())
      {
        res += s.top();
        s.pop();
      }
      reverse(res.begin() , res.end());
      // Remove the zeros from the left if they exist.
      while (res[0] == '0') res.erase(0 , 1);
    
      
      return (res == "") ? "0": res;
    }
};
```

## [Remove All Adjacent Duplicates in String II](https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string-ii)

A: 当出现相同字符，则自增检验是否达到k。

```cpp
class Solution {
public:
    string removeDuplicates(string s, int k) {
        vector<pair<int, char>> stack = {{0, '#'}};
        for (char c: s) {
            if (stack.back().second != c) {
                stack.push_back({1, c});
            } else if (++stack.back().first == k)
                stack.pop_back();
        }
        string res;
        for (auto & p : stack) {
            res.append(p.first, p.second);
        }
        return res;
    }
};
```

## [132 Pattern](https://leetcode.com/problems/132-pattern)

A: 前缀最小值数组记录1，降序栈记录3，如果不满足降序栈，则可能找到2，出栈直到发现1。

```cpp
class Solution {
public:
    bool find132pattern(vector<int>& nums) {
        stack <pair<int,int>> st;
        vector<int> prefix(nums.size(), INT_MAX);

        int mini = INT_MAX;

        for(int i = 0; i<nums.size(); i++){
            prefix[i] = mini;
            mini = min(mini, nums[i]);    
        }
        
        for(int i = 0; i<nums.size(); i++){
            if(st.empty()) st.push({nums[i],i});
            else{
                while(!st.empty() and nums[i]>=st.top().first) st.pop();

                if(!st.empty() and prefix[st.top().second]<nums[i]) return true;
                
                st.push({nums[i],i});
            }
        }
        return false;
    }
};
```

## [Implement Queue using Stacks](https://leetcode.com/problems/implement-queue-using-stacks)

A: 用输入栈和输出栈模拟队列操作。

```go
type MyQueue struct {
    s1, s2 []int
}


func Constructor() MyQueue {
    return MyQueue{[]int{}, []int{}}
}


func (this *MyQueue) Push(x int)  {
    for len(this.s2) != 0 {
        cur := this.s2[len(this.s2) - 1]
        this.s2 = this.s2[:len(this.s2) - 1]
        this.s1 = append(this.s1, cur)
    }
    this.s1 = append(this.s1, x)
}


func (this *MyQueue) Pop() int {
    for len(this.s1) != 0 {
        cur := this.s1[len(this.s1) - 1]
        this.s1 = this.s1[:len(this.s1) - 1]
        this.s2 = append(this.s2, cur)
    }
    ans := this.s2[len(this.s2) - 1]
    this.s2 = this.s2[:len(this.s2) - 1]
    return ans
}


func (this *MyQueue) Peek() int {
    for len(this.s1) != 0 {
        cur := this.s1[len(this.s1) - 1]
        this.s1 = this.s1[:len(this.s1) - 1]
        this.s2 = append(this.s2, cur)
    }
    return this.s2[len(this.s2) - 1]
}


func (this *MyQueue) Empty() bool {
    return len(this.s1) == 0 && len(this.s2) == 0
}


/**
 * Your MyQueue object will be instantiated and called as such:
 * obj := Constructor();
 * obj.Push(x);
 * param_2 := obj.Pop();
 * param_3 := obj.Peek();
 * param_4 := obj.Empty();
 */
 ```

## [Implement Stack using Queues](https://leetcode.com/problems/implement-stack-using-queues)

A: 用两个队列保证其中一个队列q1的元素顺序是栈顺序的。

```cpp
class MyStack {
public:
    /** Initialize your data structure here. */
    queue<int> q1;
    queue<int> q2;
    MyStack() {
        
    }
    
    /** Push element x onto stack. */
    void push(int x) {
        q2.push(x);
        while(!q1.empty()){
            q2.push(q1.front());    q1.pop();
        }
        swap(q1, q2);
    }
    
    /** Removes the element on top of the stack and returns that element. */
    int pop() {
        int result = top();
        q1.pop();
        return result;
    }
    
    /** Get the top element. */
    int top() {
        return q1.front();
    }
    
    /** Returns whether the stack is empty. */
    bool empty() {
        return q1.empty();
    }
};
```

A: 只用一个队列，当需要栈顶元素时，将队列中的元素依次出队并入队，直到剩下一个元素，即为栈顶元素。

```go
type MyStack struct {
    q []int
}


func Constructor() MyStack {
    return MyStack{[]int{}}
}


func (this *MyStack) Push(x int)  {
    this.q = append(this.q, x)
}


func (this *MyStack) Pop() int {
    size := len(this.q)
    for i := 0; i < size - 1; i++ {
        cur := this.q[0]
        this.q = this.q[1:]
        this.q = append(this.q, cur)
    }
    ans := this.q[0]
    this.q = this.q[1:]
    return ans
}


func (this *MyStack) Top() int {
    size := len(this.q)
    for i := 0; i < size - 1; i++ {
        cur := this.q[0]
        this.q = this.q[1:]
        this.q = append(this.q, cur)
    }
    ans := this.q[0]
    this.q = this.q[1:]
    this.q = append(this.q, ans)
    return ans
}


func (this *MyStack) Empty() bool {
    return len(this.q) == 0
}


/**
 * Your MyStack object will be instantiated and called as such:
 * obj := Constructor();
 * obj.Push(x);
 * param_2 := obj.Pop();
 * param_3 := obj.Top();
 * param_4 := obj.Empty();
 */
 ```

## [Maximum Frequency Stack](https://leetcode.com/problems/maximum-frequency-stack)

A: 记录maxFreq，建立{val, freq}和{freq, stack}的映射进行维护。

```cpp
class FreqStack {
public:
    unordered_map<int, int> freq; // val, freq
    unordered_map<int, stack<int>> m; // freq, stack
    int maxfreq = 0;

    void push(int x) {
        maxfreq = max(maxfreq, ++freq[x]);
        m[freq[x]].push(x);
    }

    int pop() {
        int x = m[maxfreq].top();
        m[maxfreq].pop();
        if (!m[freq[x]--].size()) maxfreq--;
        return x;
    }
};

/**
 * Your FreqStack object will be instantiated and called as such:
 * FreqStack* obj = new FreqStack();
 * obj->push(val);
 * int param_2 = obj->pop();
 */
```

## [Remove All Adjacent Duplicates In String](https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string)

A: 出现相同字母则出栈。

```go
func removeDuplicates(s string) string {
    stk := []rune{}
    for _, r := range s {
        if len(stk) == 0 {
            stk = append(stk, r)
            continue
        }

        top := stk[len(stk) - 1]
        if r == top {
            stk = stk[:len(stk) - 1]
        } else {
            stk = append(stk, r)
        }
    }
    return string(stk)
}
```

## [包含min函数的栈](https://leetcode.cn/problems/bao-han-minhan-shu-de-zhan-lcof)

A: 维护两个栈，一个栈作为正常栈，另一个存放当前正常栈操作对应的最小值，类似于单调栈。

```go
type MinStack struct {
    stk1 []int
    stk2 []int
}


/** initialize your data structure here. */
func Constructor() MinStack {
    return MinStack{[]int{},[]int{}}
}


func (this *MinStack) Push(x int)  {
    this.stk1 = append(this.stk1, x)
    if len(this.stk2) != 0 {
        if this.stk2[len(this.stk2)-1] < x {
            this.stk2 = append(this.stk2, this.stk2[len(this.stk2)-1])
            return
        }
    }
    this.stk2 = append(this.stk2, x)
}


func (this *MinStack) Pop()  {
    this.stk1 = this.stk1[:len(this.stk1) - 1]
    this.stk2 = this.stk2[:len(this.stk2) - 1]
}


func (this *MinStack) Top() int {
    return this.stk1[len(this.stk1) - 1]
}


func (this *MinStack) Min() int {
    return this.stk2[len(this.stk2) - 1]
}


/**
 * Your MinStack object will be instantiated and called as such:
 * obj := Constructor();
 * obj.Push(x);
 * obj.Pop();
 * param_3 := obj.Top();
 * param_4 := obj.Min();
 */
```

## [Backspace String Compare](https://leetcode.com/problems/backspace-string-compare)

A: 栈模拟。

```go
func backspaceCompare(s string, t string) bool {
    str1 := getStr(s)
    str2 := getStr(t)
    return str1 == str2
}

func getStr(s string) string {
    stk := []rune{}
    for _, ch := range s {
        if ch == '#' {
            if len(stk) > 0 {
                stk = stk[:len(stk) - 1]
            }
        } else {
            stk = append(stk, ch)
        }
    }
    return string(stk)
}
```
