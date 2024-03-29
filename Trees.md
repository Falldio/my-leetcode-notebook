# Trees

## [Binary Tree Preorder Traversal](https://leetcode.com/problems/binary-tree-preorder-traversal)

A: 用栈保存历史信息，注意入栈的顺序。

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func inorderTraversal(root *TreeNode) []int {
    stk := []*TreeNode{root}
    ans := []int{}
    if root == nil {
        return ans
    }
    for len(stk) > 0 {
        cur := stk[len(stk) - 1]
        stk = stk[:len(stk) - 1]
        if cur != nil {
            if cur.Right != nil {
                stk = append(stk, cur.Right)
            }
            stk = append(stk, cur, nil)
            if cur.Left != nil {
                stk = append(stk, cur.Left)
            }
        } else {
            cur = stk[len(stk) - 1]
            stk = stk[:len(stk) - 1]
            ans = append(ans, cur.Val)
        }
    }
    return ans
}
```

A：Morris 遍历。

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func preorderTraversal(root *TreeNode) []int {
    if root == nil {
        return []int{}
    }

    ans := []int{}
    cur := root
    var mostRight *TreeNode
    for cur != nil {
        mostRight = cur.Left
        if mostRight != nil {
            for mostRight.Right != nil && mostRight.Right != cur {
                mostRight = mostRight.Right
            }
            if mostRight.Right == nil {
                mostRight.Right = cur
                ans = append(ans, cur.Val)
                cur = cur.Left
                continue
            } else {
                mostRight.Right = nil
            }
        } else {
            ans = append(ans, cur.Val)
        }
        cur = cur.Right
    }
    return ans
}
```

## [Binary Tree Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal)

A: 使用循环时需要用栈记录历史信息，注意入栈的顺序，nil标记已经遍历但是没有加入结果的节点。

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func inorderTraversal(root *TreeNode) []int {
    stk := []*TreeNode{root}
    ans := []int{}
    if root == nil {
        return ans
    }
    for len(stk) > 0 {
        cur := stk[len(stk) - 1]
        stk = stk[:len(stk) - 1]
        if cur != nil {
            if cur.Right != nil {
                stk = append(stk, cur.Right)
            }
            stk = append(stk, cur, nil)
            if cur.Left != nil {
                stk = append(stk, cur.Left)
            }
        } else {
            cur = stk[len(stk) - 1]
            stk = stk[:len(stk) - 1]
            ans = append(ans, cur.Val)
        }
    }
    return ans
}
```

A：Morris 遍历

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func inorderTraversal(root *TreeNode) []int {
    ans := []int{}
    cur := root
    var mostRight *TreeNode
    for cur != nil {
        mostRight = cur.Left
        if mostRight != nil {
            for mostRight.Right != nil && mostRight.Right != cur {
                mostRight =mostRight.Right
            }
            if mostRight.Right == nil {
                mostRight.Right = cur
                cur = cur.Left
                continue
            } else {
                mostRight.Right = nil
                ans = append(ans, cur.Val)
            }
        } else {
            ans = append(ans, cur.Val)
        }
        cur = cur.Right
    }
    return ans
}
```

## [Binary Tree Postorder Traversal](https://leetcode.com/problems/binary-tree-postorder-traversal)

A: 用栈保存历史信息，注意入栈的顺序。

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func postorderTraversal(root *TreeNode) []int {
    ans := make([]int, 0, 100)
    stk := []*TreeNode{root}
    if root == nil {
        return ans
    }
    for len(stk) > 0 {
        cur := stk[len(stk) - 1]
        stk = stk[:len(stk) - 1]
        if cur != nil {
            stk = append(stk, cur, nil)
            if cur.Right != nil {
                stk = append(stk, cur.Right)
            }
            if cur.Left != nil {
                stk = append(stk, cur.Left)
            }
        } else {
            cur = stk[len(stk) - 1]
            stk = stk[:len(stk) - 1]
            ans = append(ans, cur.Val)
        }
    }
    return ans
}
```

## [Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree)

A: 递归交换。

```cpp
TreeNode* invertTree(TreeNode* root) {
    if (root) {
        invertTree(root->left);
        invertTree(root->right);
        std::swap(root->left, root->right);
    }
    return root;
}
```

```go
func invertTree(root *TreeNode) *TreeNode {
    if root == nil {
        return nil
    }
    root.Left, root.Right = invertTree(root.Right), invertTree(root.Left)
    return root
}
```

A: 遍历。

```cpp
TreeNode* invertTree(TreeNode* root) {
    std::stack<TreeNode*> stk;
    stk.push(root);
    
    while (!stk.empty()) {
        TreeNode* p = stk.top();
        stk.pop();
        if (p) {
            stk.push(p->left);
            stk.push(p->right);
            std::swap(p->left, p->right);
        }
    }
    return root;
}
```

## [Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree)

A: 递归。

```cpp
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if (root == nullptr) {
            return 0;
        }
        return 1 + max(maxDepth(root->left), maxDepth(root->right));
    }
};
```

A: BFS，计数时每次取出一层。

```cpp
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if (root == nullptr) {
            return 0;
        }
        queue<TreeNode*> q;
        q.push(root);
        int result = 0;
        while (!q.empty()) {
            int count = q.size();
            for (int i = 0; i < count; i++) {
                TreeNode* node = q.front();
                q.pop();
                if (node->left != nullptr) {
                    q.push(node->left);
                }
                if (node->right != nullptr) {
                    q.push(node->right);
                }
            }
            result++;
        }
        return result;
    }
};
```

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func maxDepth(root *TreeNode) int {
   if root == nil {
       return 0
   } 
   q := []*TreeNode{root}
   ans := 0
   for len(q) != 0 {
       ans++
       size := len(q)
       for i := 0; i < size; i++ {
           cur := q[0]
           q = q[1:]
           if cur.Left != nil {
               q = append(q, cur.Left)
           }
           if cur.Right != nil {
               q = append(q, cur.Right)
           }
       }
   }
   return ans
}
```

## [Same Tree](https://leetcode.com/problems/same-tree)

A: 递归比较子树。

```cpp
class Solution {
public:
    bool isSameTree(TreeNode* p, TreeNode* q) {
        if (p == nullptr && q == nullptr) {
            return true;
        }
        if (p == nullptr || q == nullptr) {
            return false;
        }
        if (p->val != q->val) {
            return false;
        }
        return isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
    }
};
```

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func isSameTree(p *TreeNode, q *TreeNode) bool {
    if p == nil && q == nil {
        return true
    }
    if p == nil || q == nil {
        return false
    }
    if p.Val != q.Val {
        return false
    }
    return isSameTree(p.Left, q.Left) && isSameTree(p.Right, q.Right)
}
```

## [Subtree of Another Tree](https://leetcode.com/problems/subtree-of-another-tree)

A: 递归比较节点值。剪枝：只比较深度相等的子树。

```cpp
class Solution {
    vector<TreeNode*> nodes;

public:
    bool isSubtree(TreeNode* s, TreeNode* t) {
        if (!s && !t) return true;
        if (!s || !t) return false;

        getDepth(s, getDepth(t, -1));

        for (TreeNode* n: nodes)
            if (identical(n, t))
                return true;

        return false;
    }

    int getDepth(TreeNode* r, int d) {
        if (!r)
            return -1;

        int depth = max(getDepth(r->left, d), getDepth(r->right, d)) + 1;

        // Check if depth equals required value
        // Require depth is -1 for tree t (only return the depth, no push)
        if (depth == d)
            nodes.push_back(r);

        return depth;
    }

    bool identical(TreeNode* a, TreeNode* b) {
        if (!a && !b) return true;
        if (!a || !b || a->val != b->val) return false;

        return identical(a->left, b->left) && identical(a->right, b->right);
    }
};
```

## [Lowest Common Ancestor of a Binary Search Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree)

A: 递归。按照大小关系搜索树。

```cpp
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if (p->val < root->val && q->val < root->val) {
            return lowestCommonAncestor(root->left, p, q);
        } else if (p->val > root->val && q->val > root->val) {
            return lowestCommonAncestor(root->right, p, q);
        } else {
            return root;
        }
    }
};
```

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val   int
 *     Left  *TreeNode
 *     Right *TreeNode
 * }
 */

func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
    if root == nil {
        return nil
    }
    if root == p || root == q {
        return root
    }
    if root.Val < p.Val && root.Val < q.Val {
        return lowestCommonAncestor(root.Right, p, q)
    } else if root.Val > p.Val && root.Val > q.Val {
        return lowestCommonAncestor(root.Left, p, q)
    } else {
        return root
    }
}
```

A: 遍历。按照大小关系搜索树。

```cpp
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        while (root != nullptr) {
            if (p->val < root->val && q->val < root->val) {
                root = root->left;
            } else if (p->val > root->val && q->val > root->val) {
                root = root->right;
            } else {
                return root;
            }
        }
        return nullptr;
    }
};
```

## [Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal)

A: 用队列记录遍历层次历史。

```cpp
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> ans;
        deque<TreeNode*> q;
        if (root)   q.push_back(root);

        while (q.size() > 0) {
            vector<int> level;
            int len = q.size();
            for (int i = 0; i < len; i++) {
                TreeNode* node = q[q.size()-1];
                level.push_back(node->val);
                q.pop_back();
                if (node->left) q.push_front(node->left);
                if (node->right)    q.push_front(node->right);
            }
            ans.push_back(level);
        }

        return ans;
    }
};
```

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func levelOrder(root *TreeNode) [][]int {
    ans := [][]int{}
    q := []*TreeNode{root}
    if root == nil {
        return ans
    }
    for len(q) > 0 {
        level := []int{}
        for i := len(q); i > 0; i-- {
            cur := q[0]
            q = q[1:]
            level = append(level, cur.Val)
            if cur.Left != nil {
                q = append(q, cur.Left)
            }
            if cur.Right != nil {
                q = append(q, cur.Right)
            }
        }
        ans = append(ans, level)
    }
    return ans
}
```

## [Binary Tree Level Order Traversal II](https://leetcode.com/problems/binary-tree-level-order-traversal-ii)

A: BFS，用队列记录遍历层次历史。每次插入到结果数组的头部。

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func levelOrderBottom(root *TreeNode) [][]int {
    ans := [][]int{}
    q := []*TreeNode{root}
    if root == nil {
        return ans
    }
    for len(q) > 0 {
        level := []int{}
        for i := len(q); i > 0; i-- {
            cur := q[0]
            q = q[1:]
            level = append(level, cur.Val)
            if cur.Left != nil {
                q = append(q, cur.Left)
            }
            if cur.Right != nil {
                q = append(q, cur.Right)
            }
        }
        ans = append(ans, []int{})
        copy(ans[1:], ans)
        ans[0] = level
    }
    return ans
}
```

## [Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree)

A: BST的中序遍历序列具有递增特性。

```cpp
class Solution {
public:
    bool isValidBST(TreeNode* root) {
        TreeNode* prev = nullptr;
        return validate(root, prev);
    }
    bool validate(TreeNode* node, TreeNode* &prev) {
        if (node == nullptr) return true;
        if (!validate(node->left, prev)) return false;
        if (prev != nullptr && prev->val >= node->val) return false;
        prev = node;
        return validate(node->right, prev);
    }
};
```

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func isValidBST(root *TreeNode) bool {
    order := []int{}
    inorder(root, &order)
    for i := 1; i < len(order); i++ {
        if order[i] <= order[i - 1] {
            return false
        }
    }
    return true
}

func inorder(root *TreeNode, order *[]int) {
    if root == nil {
        return
    }
    inorder(root.Left, order)
    *order = append(*order, root.Val)
    inorder(root.Right, order)
}
```

## [Kth Smallest Element in a BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst)

A: 中序遍历，找到第k个元素。

```cpp
class Solution {
public:
    int ans;
    
    int kthSmallest(TreeNode* root, int k) {
        inorder(root, k);
        return ans;
    }
    
    void inorder(TreeNode* root, int& k) {
        if (!root) return;
        inorder(root->left, k);
        if (--k == 0){
            ans = root->val;
            return;
        } 
        inorder(root->right, k);
    }  
};
```

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
var (
    ans []int
)
func kthSmallest(root *TreeNode, k int) int {
    ans = []int{}
    traverse(root)
    return ans[k - 1]
}

func traverse(root *TreeNode) {
    if root == nil {
        return
    }
    traverse(root.Left)
    ans = append(ans, root.Val)
    traverse(root.Right)
}
```

## [Construct Binary Tree from Preorder and Inorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal)

A: 前序遍历当前的第一个元素始终是根节点，该元素在中序遍历中切分了左右子树。

```cpp
class Solution {
public:
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        int index = 0;
        return build(preorder, inorder, index, 0, inorder.size() - 1);
    }
private:
    TreeNode* build(vector<int>& preorder, vector<int>& inorder, int& index, int i, int j) {
        // index: 当前节点在preorder中的位置。
        // i和j: 标识inorder中的子树片段。
        if (i > j) {
            return nullptr;
        }
        
        TreeNode* root = new TreeNode(preorder[index]);
        
        int split = 0;
        for (int i = 0; i < inorder.size(); i++) {
            if (preorder[index] == inorder[i]) {
                split = i;
                break;
            }
        }
        index++;
        
        root->left = build(preorder, inorder, index, i, split - 1);
        root->right = build(preorder, inorder, index, split + 1, j);
        
        return root;
    }
};
```

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func buildTree(preorder []int, inorder []int) *TreeNode {
    if len(preorder) == 0 {
        return nil
    }
    root := &TreeNode{Val: preorder[0]}
    mid := 0
    for k, v := range inorder {
        if v == root.Val {
            mid = k
            break
        }
    }
    root.Left = buildTree(preorder[1:mid+1], inorder[:mid])
    root.Right = buildTree(preorder[mid+1:], inorder[mid+1:])
    return root
}
```

## [Construct Binary Tree from Inorder and Postorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal)

A: 后序遍历的最后一个元素是根节点，该元素在中序遍历中切分了左右子树，以此确定左右子树节点数目。

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func buildTree(inorder []int, postorder []int) *TreeNode {
    if len(inorder) == 0 {
        return nil
    }
    root := &TreeNode{Val: postorder[len(postorder) - 1]}
    numLeft := 0
    for k, v := range inorder {
        if v == root.Val {
            numLeft = k
            break
        }
    }
    root.Left = buildTree(inorder[:numLeft], postorder[:numLeft])
    root.Right = buildTree(inorder[numLeft+1:], postorder[numLeft:len(postorder)-1])
    return root
}
```

## [Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum)

A: DFS，遍历时计算当前节点子树的最大和。

```cpp
class Solution {
public:
    int maxPathSum(TreeNode* root) {
        int maxPath = INT_MIN;
        dfs(root, maxPath);
        return maxPath;
    }
private:
    int dfs(TreeNode* root, int& maxPath) {
        if (root == nullptr) {
            return 0;
        }
        
        int left = max(dfs(root->left, maxPath), 0);
        int right = max(dfs(root->right, maxPath), 0);
        
        // maxPath取舍：取一个子树加父路径或舍父路径取两个子树。
        int curPath = root->val + left + right;
        maxPath = max(maxPath, curPath);
        
        // 返回时只返回当前节点下单一子树最大值，便于累加。
        return root->val + max(left, right);
    }
};
```

## [Serialize and Deserialize Binary Tree](https://leetcode.com/problems/serialize-and-deserialize-binary-tree)

A:

+ serialize: 前序遍历，标记空节点。
+ deserialize: 递归解码。

```cpp
class Codec {
public:

    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
        ostringstream out;
        encode(root, out);
        return out.str();
    }

    // Decodes your encoded data to tree.
    TreeNode* deserialize(string data) {
        istringstream in(data);
        return decode(in);
    }
    
private:
    
    void encode(TreeNode* root, ostringstream& out) {
        if (root == NULL) {
            out << "N ";
            return;
        }
        
        out << root->val << " ";
        
        encode(root->left, out);
        encode(root->right, out);
    }
    
    TreeNode* decode(istringstream& in) {
        string value = "";
        in >> value;
        
        if (value == "N") {
            return nullptr;
        }
        
        TreeNode* root = new TreeNode(stoi(value));
        
        root->left = decode(in);
        root->right = decode(in);
        
        return root;
    }
    
};
```

```go
func Serialize(root *TreeNode) string {
	ans := []byte{}
	encode(root, &ans)
	return string(ans)
}

func encode(root *TreeNode, ans *[]byte) {
	if root == nil {
		*ans = append(*ans, "$,"...)
	} else {
		*ans = append(*ans, fmt.Sprintf("%d,", root.Val)...)
		encode(root.Left, ans)
		encode(root.Right, ans)
	}
}

func Deserialize(s string) *TreeNode {
    strs := strings.Split(s, ",")
    return decode(&strs)
}

func decode(strs *[]string) *TreeNode {
    if len(*strs) == 0 {
        return nil
    }
    if (*strs)[0] == "$" {
        *strs = (*strs)[1:]
        return nil
    }
    val, _ := strconv.Atoi((*strs)[0])
    *strs = (*strs)[1:]
    node := &TreeNode{Val: val}
    node.Left = decode(strs)
    node.Right = decode(strs)
    return node
}
```

A: Golang json解码。

```go
type Codec struct {
    
}

func Constructor() Codec {
    codec := Codec{}
    return codec
}

// Serializes a tree to a single string.
func (this *Codec) serialize(root *TreeNode) string {
    if nil == root {
        return "[]"
    }
    b, _ := json.Marshal(root)
    return string(b)
}

// Deserializes your encoded data to tree.
func (this *Codec) deserialize(data string) *TreeNode {
    if "[]" == data {
        return nil
    }
    var root TreeNode
    json.Unmarshal([]byte(data), &root)
    return &root
}
```

## [Diameter of Binary Tree](https://leetcode.com/problems/diameter-of-binary-tree)

A: 后序遍历，计算左右子树最大深度之和。

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
int diameterOfBinaryTree(TreeNode* root) {
        int d = 0;
        rec(root, d);
        return d;
    }
    
    int rec(TreeNode* root, int &d) {
        if(root == nullptr) return 0;
        int ld = rec(root->left, d);
        int rd = rec(root->right, d);
        d = max(d, ld + rd);
        return max(ld, rd) + 1;
    }
};
```

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
var (
    ans int
)

func diameterOfBinaryTree(root *TreeNode) int {
    ans = 0
    dfs(root)
    return ans
}

func dfs(root *TreeNode) int {
    if root == nil {
        return 0
    }
    left := dfs(root.Left)
    right := dfs(root.Right)
    d := left + right
    ans = max(ans, d)
    return max(left, right) + 1
}

func max(i, j int) int {
    if i < j {
        return j
    } else {
        return i
    }
}
```

## [Balanced Binary Tree](https://leetcode.com/problems/balanced-binary-tree)

A: 计算左右子树高度再相减进行判断。

```cpp
class Solution {
public:
    bool isBalanced(TreeNode* root) {
        int height = 0;
        return dfs(root, height);
    }
private:
    bool dfs(TreeNode* root, int& height) {
        if (root == NULL) {
            height = -1;
            return true;
        }
        
        int left = 0;
        int right = 0;
        
        if (!dfs(root->left, left) || !dfs(root->right, right)) {
            return false;
        }
        if (abs(left - right) > 1) {
            return false;
        }
        
        height = 1 + max(left, right);
        return true;
    }
};
```

```go
func isBalanced(root *TreeNode) bool {
    h := getHeight(root) 
    if h == -1 {
        return false
    }
    return true
}
// 返回以该节点为根节点的二叉树的高度，如果不是平衡二叉树了则返回-1
func getHeight(root *TreeNode) int {
    if root == nil {
        return 0
    }
    l, r := getHeight(root.Left), getHeight(root.Right)
    if l == -1 || r == -1 {
        return -1
    }
    if l - r > 1 || r - l > 1 {
        return -1
    }
    return max(l, r) + 1
}
func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

## [Binary Tree Right Side View](https://leetcode.com/problems/binary-tree-right-side-view)

A: 先右后左dfs，用`level`控制条件判断。

```cpp
class Solution {
public:
    vector<int> rightSideView(TreeNode *root) {
        vector<int> res;
        recursion(root, 1, res);
        return res;
    }
    
    void recursion(TreeNode *root, int level, vector<int> &res)
    {
        if(root == NULL) return ;
        if(res.size() < level) res.push_back(root->val);
        recursion(root->right, level+1, res);
        recursion(root->left, level+1, res);
    }
};
```

A: BFS。

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func rightSideView(root *TreeNode) []int {
    ans := []int{}
    q := []*TreeNode{root}
    if root == nil {
        return ans
    }
    for len(q) > 0 {
        ans = append(ans, q[len(q) - 1].Val)
        for i := len(q); i > 0; i-- {
            cur := q[0]
            q = q[1:]
            if cur.Left != nil {
                q = append(q, cur.Left)
            }
            if cur.Right != nil {
                q = append(q, cur.Right)
            }
        }
    }
    return ans
}
```

## [Count Good Nodes In Binary Tree](https://leetcode.com/problems/count-good-nodes-in-binary-tree)

A: 遍历的同时保留路径中的最大节点。

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    int curMax = INT_MIN;
    int goodNodes(TreeNode* root) {
        int ans = 0;
        dfs(ans, root);
        return ans;
    }
private:
    void dfs(int &ans, TreeNode * node) {
        if (node) {
            int oldMax = curMax;
            if (node->val >= curMax) {
                curMax = node->val;
                ans++;
            }
            dfs(ans, node->left);
            dfs(ans, node->right);
            curMax= oldMax;
        }
    }
};
```

## [Minimum Time to Collect All Apples in a Tree](https://leetcode.com/problems/minimum-time-to-collect-all-apples-in-a-tree)

A: DFS，传入前一个结点防止来回无限遍历。

```cpp
class Solution {
public:
    vector<vector<int>> adjList;
    int dfs(vector<bool>& hasApple,int node,int d,int prev) {
        int result = 0, temp;
        for (int &i : adjList[node]) {
            if (i != prev) {
                temp = dfs(hasApple, i, d + 1, node);
                if (temp) // If child has apples it'll return a non zero result which is the distance traveled upto that node.
                    result += temp - d;
            }
        }
        return result || hasApple[node] ? result + d : 0;  // If nothing is added to result and current node doesnt have apple return 0 else return distances of children + current distance from root.
        
    }
    int minTime(int n, vector<vector<int>>& edges, vector<bool>& hasApple) {
        adjList.resize(n);
        for(vector<int> &e:edges)
            adjList[e[0]].push_back(e[1]),adjList[e[1]].push_back(e[0]);
        return dfs(hasApple, 0, 0, -1) * 2;     // Result is doubled the distance travelled as per our observation.
    }
};
```

## [Number of Nodes in the Sub-Tree With the Same Label](https://leetcode.com/problems/number-of-nodes-in-the-sub-tree-with-the-same-label)

A: DP+DFS。

```cpp
class Solution {
public:
    vector<int> fun(vector<vector<int>> &adj, string &labels, int i,vector<int>&result){
        vector<int> ans(26, 0); // ans记录当前子树的所有字母数量
        result[i] = 1;
        ans[labels[i] - 'a'] = 1;
    
        for (int j = 0; j != adj[i].size(); j++)
            if (!result[adj[i][j]]) {
                vector<int> tmp = fun(adj, labels,adj[i][j],result);
                for(int k = 0; k != 26; k++) ans[k] += tmp[k];
            }
    
        result[i] = ans[labels[i] - 'a'];
    
        return ans;
    }
    
    vector<int> countSubTrees(int n, vector<vector<int>>& edges, string labels) {
        vector<vector<int>> adj(n);
        vector<int> result(n,0);
        for(int i = 0; i != edges.size(); i++) {
            adj[edges[i][0]].push_back(edges[i][1]);
            adj[edges[i][1]].push_back(edges[i][0]);
        }
    
        fun(adj, labels, 0, result);
        return result;
    }
};
```

## [Longest Path With Different Adjacent Characters](https://leetcode.com/problems/longest-path-with-different-adjacent-characters)

A: DFS，返回从当前节点i开始的最长路径。

```cpp
class Solution {
public:
    int longestPath(vector<int>& parent, string s) {
        int n = s.size(), res = 0;
        vector<vector<int>> children(n, vector<int>());
        for (int i = 1; i < n; ++i)
            children[parent[i]].push_back(i);
        dfs(children, s, res, 0);
        return res;
    }

    int dfs(vector<vector<int>>& children, string& s, int& res, int i) {
        int big1 = 0, big2 = 0;
        for (int& j : children[i]) {
            int cur = dfs(children, s, res, j);
            if (s[i] == s[j]) continue;
            if (cur > big2) big2 = cur;
            if (big2 > big1) swap(big1, big2);
        }
        res = max(res, big1 + big2 + 1); // 取两个子树
        return big1 + 1; //取最长子树
    }
};
```

## [Number of Good Paths](https://leetcode.com/problems/number-of-good-paths)

A: 存储相同值所有节点、邻接表只存储符合good path要求的结点，并查集确定结果。

[详解](https://leetcode.com/problems/number-of-good-paths/solutions/2621772/c-java-diagram-union-find/?orderBy=most_votes)

```cpp
class UnionFind {
    private:
        vector<int> id, rank;
        int cnt;
    public:
        UnionFind(int cnt) : cnt(cnt) {
            id = vector<int>(cnt);
            rank = vector<int>(cnt, 0);
            for (int i = 0; i < cnt; ++i) id[i] = i;
        }
        int find(int p) {
            if (id[p] == p) return p;
            return id[p] = find(id[p]);
        }
        bool connected(int p, int q) { 
            return find(p) == find(q); 
        }
        void connect(int p, int q) {
            int i = find(p), j = find(q);
            if (i == j) return;
            if (rank[i] < rank[j]) {
                id[i] = j;  
            } else {
                id[j] = i;
                if (rank[i] == rank[j]) rank[j]++;
            }
            --cnt;
        }
};

class Solution {
public:
    int numberOfGoodPaths(vector<int>& vals, vector<vector<int>>& edges) {
        int N = vals.size(), goodPaths = 0;
        vector<vector<int>> adj(N);
        map<int, vector<int>> sameValues; // map升序排列
        
        for (int i = 0; i < N; i++) {
            sameValues[vals[i]].push_back(i);
        }
        
        for (auto &e : edges) {
            int u = e[0], v = e[1];
            
            if (vals[u] >= vals[v]) {
                adj[u].push_back(v);
            } else if (vals[v] >= vals[u]) {
                adj[v].push_back(u);
            }
        }
        
        UnionFind uf(N);
        
        for (auto &[value, allNodes] : sameValues) {
            
            for (int u : allNodes) {
                for (int v : adj[u]) {
                    uf.connect(u, v);
                }
            }
            
            unordered_map<int, int> group;
            
            for (int u : allNodes) {
                group[uf.find(u)]++;
            }
            
            goodPaths += allNodes.size();
            
            for (auto &[_, size] : group) {
                goodPaths += (size * (size - 1) / 2);
            }
        }
        
        return goodPaths;
    }
};
```

## [Populating Next Right Pointers In Each Node](https://leetcode.com/problems/populating-next-right-pointers-in-each-node)

A: 层序遍历。

```cpp
/*
// Definition for a Node.
class Node {
public:
    int val;
    Node* left;
    Node* right;
    Node* next;

    Node() : val(0), left(NULL), right(NULL), next(NULL) {}

    Node(int _val) : val(_val), left(NULL), right(NULL), next(NULL) {}

    Node(int _val, Node* _left, Node* _right, Node* _next)
        : val(_val), left(_left), right(_right), next(_next) {}
};
*/

class Solution {
public:
    Node* connect(Node* root) {
        if (!root) return root;
        deque<Node *> dq;
        Node *cur = root, *pre = nullptr;
        dq.push_back(cur);
        int cnt = 1;
        while (!dq.empty()) {
            int next = 0;
            while (cnt--) {
                cur = dq.front();
                cur->next = pre;
                if (cur->left) {
                    dq.push_back(cur->right);
                    dq.push_back(cur->left);
                    next += 2;
                }
                dq.pop_front();
                pre = cur;
            }
            pre = nullptr;
            cnt = next;
        }
        return root;
    }
};
```

```go
/**
 * Definition for a Node.
 * type Node struct {
 *     Val int
 *     Left *Node
 *     Right *Node
 *     Next *Node
 * }
 */

func connect(root *Node) *Node {
    if root == nil {
        return root
    }
    cnt := 1
    dq := []*Node {root}
    var pre *Node = nil
    var cur *Node = nil
    for len(dq) > 0 {
        next := 0
        for ; cnt > 0; cnt-- {
            cur = dq[0]
            cur.Next = pre
            if cur.Left != nil {
                dq = append(dq, cur.Right)
                dq = append(dq, cur.Left)
                next += 2
            }
            dq = dq[1:]
            pre = cur
        }
        cnt = next
        pre = nil
    }
    return root
}
```

A：前序遍历，常量级别空间复杂度。

```go
/**
 * Definition for a Node.
 * type Node struct {
 *     Val int
 *     Left *Node
 *     Right *Node
 *     Next *Node
 * }
 */

func connect(root *Node) *Node {
    if root == nil {
        return nil
    }
    if root.Left != nil {
        root.Left.Next = root.Right
    }
    if root.Right != nil && root.Next != nil {
        root.Right.Next = root.Next.Left
    }
    connect(root.Left)
    connect(root.Right)
    return root
}
```

## [Populating Next Right Pointers In Each Node II](https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii)

A: BFS。

```go
/**
 * Definition for a Node.
 * type Node struct {
 *     Val int
 *     Left *Node
 *     Right *Node
 *     Next *Node
 * }
 */

func connect(root *Node) *Node {
    if root == nil {
        return nil
    }
    q := []*Node{root}
    for len(q) > 0 {
        var next *Node
        for i := len(q); i > 0; i-- {
            cur := q[0]
            q = q[1:]
            cur.Next = next
            next = cur
            if cur.Right != nil {
                q = append(q, cur.Right)
            }
            if cur.Left != nil {
                q = append(q, cur.Left)
            }
        }
    }
    return root
}
```

## [Unique Binary Search Trees](https://leetcode.com/problems/unique-binary-search-trees)

[详解](https://leetcode.com/problems/unique-binary-search-trees/solutions/1565543/c-python-5-easy-solutions-w-explanation-optimization-from-brute-force-to-dp-to-catalan-o-n/?orderBy=most_votes)

A: 组合问题。

```cpp
class Solution {
public:
    int numTrees(int n) {
        long ans = 1;
        for(int i = 1; i < n; i++)  
            ans = ans*(n+i+1) / i;    // do note that numerator and denominator will always be divisible
        return ans / n;
    }
};
```

A: DP。

```cpp
class Solution {
public:
    int numTrees(int n) {
        vector<int> dp(n+1);
        dp[0] = dp[1] = 1;
        for(int i = 2; i <= n; i++) 
            for(int j = 1; j <= i; j++)
                dp[i] += dp[j-1] * dp[i-j];
        return dp[n];
    }
};
```

```go
func numTrees(n int) int {
    dp := make([]int, n + 1)
    dp[0] = 1
    dp[1] = 1
    for i := 2; i < n + 1; i++ {
        for j := 0; j < i; j++ {
            dp[i] += dp[j] * dp[i - 1 - j]
        }
    }
    return dp[n]
}
```

## [Sum Root to Leaf Numbers](https://leetcode.com/problems/sum-root-to-leaf-numbers)

A: DFS。

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    int sumNumbers(TreeNode* root) {
        int ans = 0;
        dfs(root, ans, 0);
        return ans;
    }
private:
    void dfs(TreeNode *node, int &ans, int cur) {
        if (!node) return;
        if (!node->left && !node->right) {
            ans += 10 * cur + node->val;
            return;
        }
        dfs(node->left, ans, cur * 10 + node->val);
        dfs(node->right, ans, cur * 10 + node->val);
    }
};
```

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
var (
    ans int
)

func sumNumbers(root *TreeNode) int {
    ans = 0
    if root == nil {
        return 0
    }
    dfs(root, 0)
    return ans
}

func dfs(root *TreeNode, cur int) {
    cur = cur * 10 + root.Val
    if root.Left == nil && root.Right == nil {
        fmt.Println(ans, cur)
        ans += cur
        return
    }
    if root.Left != nil {
        dfs(root.Left, cur)
    }
    if root.Right != nil {
        dfs(root.Right, cur)
    }
}
```

## [House Robber III](https://leetcode.com/problems/house-robber-iii)

A: pair<int, int>保存是否rob当前节点。

```cpp
class Solution {
public:
    int rob(TreeNode* root) {
        auto ans = dfs(root);
        return max(ans.first, ans.second);
    }
    pair<int, int> dfs(TreeNode* root) {
        if(!root) return {0, 0};
        auto [leftDontRob, leftRob]   = dfs(root -> left);
        auto [rightDontRob, rightRob] = dfs(root -> right);
        return {
            max(leftDontRob, leftRob) + max(rightDontRob, rightRob),
            root -> val + leftDontRob + rightDontRob
        };
    }
};
```

A: DFS的多返回值包含是否计算当前节点。

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func rob(root *TreeNode) int {
    return max(dfs(root))
}

func dfs(root *TreeNode) (int, int) {
    if root == nil {
        return 0, 0
    }
    left0, left1 := dfs(root.Left)
    right0, right1 := dfs(root.Right)
    return root.Val + left1 + right1, max(left1 + right1, left0 + right0, left1 + right0, left0 + right1)
}

func max(nums ...int) int {
    cur := -1
    for _, n := range nums {
        if n > cur {
            cur = n
        }
    }
    return cur
}
```

## [Flip Equivalent Binary Trees](https://leetcode.com/problems/flip-equivalent-binary-trees)

A: 按照情况递归比较。

```cpp
class Solution {
public:
    bool flipEquiv(TreeNode* root1, TreeNode* root2) {
        // Two null trees are flip equivalent
        // A non-null and null tree are NOT flip equivalent
        // Two non-null trees with different root values are NOT flip equivalent
        // Two non-null trees are flip equivalent if
        //      The left subtree of tree1 is flip equivalent with the left subtree of tree2 and the right subtree of tree1 is   
        //      flipequivalent with the right subtree of tree2 (no flip case)
        //      OR
        //      The right subtree of tree1 is flip equivalent with the left subtree of tree2 and the left subtree of tree1 is
        //      flipequivalent with the right subtree of tree2 (flip case)
        if ( !root1 && !root2 ) return true;
        if ( !root1 && root2 || root1 &&!root2 || root1->val != root2->val ) return false;
        return flipEquiv( root1->left, root2->left ) && flipEquiv( root1->right, root2->right )
            || flipEquiv( root1->right, root2->left ) && flipEquiv( root1->left, root2->right );
    }
};
```

## [Operations on Tree](https://leetcode.com/problems/operations-on-tree)

A: 用DFS检查祖先和后代，并解锁后代。

```cpp
class LockingTree {
public:
    unordered_map<int, vector<int>> descendents;
    vector<vector<int>> Node;
    /*
        Node[i][0] = parent[i]
        Node[i][1] = -1; (means unlocked)
        Node[i][1] = x;  (means locked by user x)
    */
    int n;
    LockingTree(vector<int>& parent) {
        n = parent.size();
        Node.resize(n, vector<int>(2, -1));
        
        Node[0][0] = -1; //root has no parent
        for(int i = 1; i<n; i++) {
            Node[i][0] = parent[i];
            descendents[parent[i]].push_back(i);            
        }
    }
    
    bool lock(int num, int user) {
        if(Node[num][1] != -1) return false;
        
        Node[num][1] = user;
        return true;
    }
    
    bool unlock(int num, int user) {
        if(Node[num][1] != user) return false;
        
        Node[num][1] = -1;
        return true;
    }
    
    //Condition-2 (Atleast one descendent should be locked)
    void checkDescendents(int num, bool& atleastOne) {
        if(descendents.count(num) == 0 || descendents[num].size() == 0)
            return;
        
        for(int& x : descendents[num]) {
            if(Node[x][1] != -1) {
                atleastOne = true;
                return;
            }
            checkDescendents(x, atleastOne);
        }
    }
    
    //Condition-3 (Check if any ancestor is locked)
    bool IsAnyAncestorLocked(int& num) {
        if(num == -1)
            return false; //you reached end and found none locked
        
        return Node[num][1] != -1 || IsAnyAncestorLocked(Node[num][0]);
    }
    
    void unlockDescendents(int num) {
        if(descendents.count(num) == 0 || descendents[num].size() == 0)
            return;
        
        for(int& x : descendents[num]) {
            Node[x][1] = -1;
            unlockDescendents(x);
        }
    }
    
    bool upgrade(int num, int user) {
        //condition : 1
        if(Node[num][1] != -1) return false;
        
        
        //condition : 2
        bool atleastOne = false;
        checkDescendents(num, atleastOne);
        //If no node was locked, return false
        if(!atleastOne) return false;
        
        
        //condition : 3
        if(IsAnyAncestorLocked(Node[num][0])) return false;
        
        
        //Do the rest
        unlockDescendents(num);
        Node[num][1] = user;
        return true;
    }
};
```

## [All Possible Full Binary Trees](https://leetcode.com/problems/all-possible-full-binary-trees)

A: DFS。

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    TreeNode* clone(TreeNode* root) {
        TreeNode* new_root = new TreeNode(0);
        new_root->left = (root->left) ? clone(root->left) : nullptr;
        new_root->right = (root->right) ? clone(root->right) : nullptr; 
        return new_root;
    } 

    vector<TreeNode*> allPossibleFBT(int N) {
        std::vector<TreeNode*> ret;
        if (1 == N) {
            ret.emplace_back(new TreeNode(0));
        } else if (N % 2) {
            for (int i = 2; i <= N; i += 2) {
                auto left = allPossibleFBT(i - 1); // 分配左右子树结点数目
                auto right = allPossibleFBT(N - i);
                for (int l_idx = 0; l_idx < left.size(); ++l_idx) {
                    for (int r_idx = 0; r_idx < right.size(); ++r_idx) {
                        ret.emplace_back(new TreeNode(0));  
                        // If we're using the last right branch, then this will be the last time this left branch is used and can hence
                        // be shallow copied, otherwise the tree will have to be cloned
                        ret.back()->left = (r_idx == right.size() - 1) ? left[l_idx] : clone(left[l_idx]);  
                        // If we're using the last left branch, then this will be the last time this right branch is used and can hence
                        // be shallow copied, otherwise the tree will have to be cloned
                        ret.back()->right = (l_idx == left.size() - 1) ? right[r_idx] : clone(right[r_idx]);
                    }
                }
            }
        }
        return ret;
    }
};
```

## [Find Bottom Left Tree Value](https://leetcode.com/problems/find-bottom-left-tree-value)

A: 层序遍历取最左侧值。

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    int findBottomLeftValue(TreeNode* root) {
        queue<TreeNode *> q;
        q.push(root);
        int ans = root->val;
        while (!q.empty()) {
            int cnt = q.size();
            ans = q.front()->val;
            for (int i = 0; i < cnt; i++) {
                TreeNode *node = q.front();
                q.pop();
                if (node->left) q.push(node->left);
                if (node->right) q.push(node->right);
            }
        }
        return ans;
    }
};
```

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func findBottomLeftValue(root *TreeNode) int {
    q := []*TreeNode{root}
    var ans int
    for len(q) > 0 {
        ans = q[0].Val
        for i := len(q); i > 0; i-- {
            cur := q[0]
            q = q[1:]
            if cur.Left != nil {
                q = append(q, cur.Left)
            }
            if cur.Right != nil {
                q = append(q, cur.Right)
            }
        }
    }
    return ans
}
```

## [Trim a Binary Search Tree](https://leetcode.com/problems/trim-a-binary-search-tree)

A: DFS，dummy root。

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    TreeNode* trimBST(TreeNode* root, int low, int high) {
        TreeNode* dummy = new TreeNode(-1);
        dummy->right = root;
        dfs(root, dummy, low, high);
        return dummy->right;
    }
private:
    void dfs(TreeNode* cur, TreeNode* pre, int low, int high) {
        if (!cur) return;
        if (cur->val > high) {
            pre->right = cur->left;
            dfs(cur->left, pre, low, high);
        }
        else if (cur->val < low) {
            if (pre->val != -1) pre->left = cur->right;
            else pre->right = cur->right;
            dfs(cur->right, pre, low, high);
        } else {
            dfs(cur->left, cur, low, high);
            dfs(cur->right, cur, low, high);
        }
    }
};
```

A: DFS，注意子树也需要trim。

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func trimBST(root *TreeNode, low int, high int) *TreeNode {
    if root == nil {
        return nil
    }
    if root.Val < low  {
        return trimBST(root.Right, low, high)
    }
    if root.Val > high {
        return trimBST(root.Left, low, high)
    }
    root.Left = trimBST(root.Left, low, high)
    root.Right = trimBST(root.Right, low, high)
    return root
}
```

## [Binary Search Tree Iterator](https://leetcode.com/problems/binary-search-tree-iterator)

A: 用栈存放左子节点，每当出现移动，则将栈顶右节点入栈。

```cpp
class BSTIterator {
    stack<TreeNode *> myStack;
public:
    BSTIterator(TreeNode *root) {
        pushAll(root);
    }

    /** @return whether we have a next smallest number */
    bool hasNext() {
        return !myStack.empty();
    }

    /** @return the next smallest number */
    int next() {
        TreeNode *tmpNode = myStack.top();
        myStack.pop();
        pushAll(tmpNode->right);
        return tmpNode->val;
    }

private:
    void pushAll(TreeNode *node) {
        for (; node != NULL; myStack.push(node), node = node->left);
    }
};
```

## [Convert BST to Greater Tree](https://leetcode.com/problems/convert-bst-to-greater-tree)

A: 中序遍历的变体，优先遍历右侧最大值的结点，获取curSum。

```cpp
class Solution {
private:
    int cur_sum = 0;
public:
    void travel(TreeNode* root){
        if (!root) return;
        if (root->right) travel(root->right);
        
        root->val = (cur_sum += root->val);
        if (root->left) travel(root->left);
    }
    TreeNode* convertBST(TreeNode* root) {
        travel(root);
        return root;
    }
};
```

```go
var pre int
func convertBST(root *TreeNode) *TreeNode {
    pre = 0
    traversal(root)
    return root
}

func traversal(cur *TreeNode) {
    if cur == nil {
        return
    }
    traversal(cur.Right)
    cur.Val += pre
    pre = cur.Val
    traversal(cur.Left)
}
```

## [Insert into a Binary Search Tree](https://leetcode.com/problems/insert-into-a-binary-search-tree)

A: 递归。

```cpp
class Solution {
public:
    TreeNode* insertIntoBST(TreeNode* root, int val) {
        if(!root) return new TreeNode(val);
        if(root->val > val) root->left = insertIntoBST(root->left, val);
        else root->right = insertIntoBST(root->right, val);
        return root;
    }
};
```

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func insertIntoBST(root *TreeNode, val int) *TreeNode {
    if root == nil {
        return &TreeNode{Val: val}
    }
    if root.Val > val {
        dfs(root.Left, root, val)
    } else {
        dfs(root.Right, root, val)
    }
    return root
}

func dfs(node, pre *TreeNode, val int) {
    if node == nil {
        if pre.Val > val {
            pre.Left = &TreeNode{Val: val}
        } else {
            pre.Right = &TreeNode{Val: val}
        }
        return
    }
    if node.Val > val {
        dfs(node.Left, node, val)
    } else {
        dfs(node.Right, node, val)
    }
}
```

## [Delete Node in a BST](https://leetcode.com/problems/delete-node-in-a-bst)

A: 分多种情况讨论，当删除的结点有两个子树时，需要用左子树的右下角或右子树的左下角替换，注意这种情况下的while循环。

```cpp
class Solution {
public:
    TreeNode* deleteNode(TreeNode* root, int key) {
        if(root) 
            if(key < root->val) root->left = deleteNode(root->left, key);     //We frecursively call the function until we find the target node
            else if(key > root->val) root->right = deleteNode(root->right, key);       
            else{
                if(!root->left && !root->right) return NULL;          //No child condition
                if (!root->left || !root->right)
                    return root->left ? root->left : root->right;    //One child contion -> replace the node with it's child
                                                                    //Two child condition   
                TreeNode* temp = root->left;                        //(or) TreeNode *temp = root->right;
                while(temp->right != NULL) temp = temp->right;     //      while(temp->left != NULL) temp = temp->left;
                root->val = temp->val;                            //       root->val = temp->val;
                root->left = deleteNode(root->left, temp->val);  //        root->right = deleteNode(root->right, temp);
            }
        return root;
    }   
};
```

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func deleteNode(root *TreeNode, key int) *TreeNode {
    if root == nil {
        return nil
    }
    if root.Val == key {
        if root.Left == nil && root.Right == nil {
            return nil
        } else if root.Left == nil {
            return root.Right
        } else if root.Right == nil {
            return root.Left
        } else {
            cur := root.Left
            for cur.Right != nil {
                cur = cur.Right
            }
            cur.Right = root.Right
            return root.Left
        }
    } else if root.Val > key {
        root.Left = deleteNode(root.Left, key)
        return root
    } else {
        root.Right = deleteNode(root.Right, key)
        return root
    }
}
```

## [Binary Tree Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal)

A: 中序遍历。

```cpp
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> nodes;
        inorder(root, nodes);
        return nodes;
    }
private:
    void inorder(TreeNode* root, vector<int>& nodes) {
        if (!root) {
            return;
        }
        inorder(root -> left, nodes);
        nodes.push_back(root -> val);
        inorder(root -> right, nodes);
    }
};
```

## [Convert Sorted Array to Binary Search Tree](https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree)

A: 给定中序遍历，构建BST，分步骤找中点即可。

```cpp
class Solution {
public:
    TreeNode* sortedArrayToBST(vector<int>& nums) {
        return helper(nums, 0, nums.size()-1);
    }
    
    TreeNode* helper(vector<int>& nums, int left, int right){
        
        //base case
        //If the left pointer crosses right return null;
        if(left > right){
            return NULL;
        }
        
        // as middle of the array will be the root node
        int mid = (left + right) / 2;
        TreeNode* temp = new TreeNode(nums[mid]);
        
        //left part from middle will be left subtree
        temp->left = helper(nums, left, mid-1);
        
        //right part of array will be right subtree
        temp->right = helper(nums, mid+1 , right);
        return temp;
    }
};
```

```go
func sortedArrayToBST(nums []int) *TreeNode {
    if len(nums) == 0 {    //终止条件，最后数组为空则可以返回
        return nil
    }
    idx := len(nums)/2
    root := &TreeNode{Val: nums[idx]} 
     
    root.Left = sortedArrayToBST(nums[:idx])
    root.Right = sortedArrayToBST(nums[idx+1:])

    return root
}
```

## [Merge Two Binary Trees](https://leetcode.com/problems/merge-two-binary-trees)

A: DFS，两棵树都非空则返回两数之和，反之返回非空节点。

```cpp
class Solution {
public:
    TreeNode* mergeTrees(TreeNode* t1, TreeNode* t2) {
        if ( t1 && t2 ) {
            TreeNode * root = new TreeNode(t1->val + t2->val);
            root->left = mergeTrees(t1->left, t2->left);
            root->right = mergeTrees(t1->right, t2->right);
            return root;
        } else {
            return t1 ? t1 : t2;
        }
    }
};
```

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func mergeTrees(root1 *TreeNode, root2 *TreeNode) *TreeNode {
    if root1 != nil && root2 != nil {
        root1.Val += root2.Val
        root1.Left = mergeTrees(root1.Left, root2.Left)
        root1.Right = mergeTrees(root1.Right, root2.Right)
        return root1
    }
    if root1 != nil {
        return root1
    } else {
        return root2
    }
}
```

## [Path Sum](https://leetcode.com/problems/path-sum)

A: DFS。

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    bool hasPathSum(TreeNode *root, int sum) {
        if (root == NULL) return false;
        if (root->val == sum && root->left ==  NULL && root->right == NULL) return true;
        return hasPathSum(root->left, sum-root->val) || hasPathSum(root->right, sum-root->val);
    }
};
```

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func hasPathSum(root *TreeNode, targetSum int) bool {
    sum := 0
    if root == nil {
        return false
    }
    return dfs(root, sum, targetSum)
}

func dfs(root *TreeNode, sum, target int) bool {
    if root.Left == nil && root.Right == nil {
        return sum + root.Val == target
    }
    ans := false
    if root.Left != nil {
        ans = ans || dfs(root.Left, sum + root.Val, target)
    }
    if root.Right != nil {
        ans = ans || dfs(root.Right, sum + root.Val, target)
    }
    return ans
}
```

## [Construct String from Binary Tree](https://leetcode.com/problems/construct-string-from-binary-tree)

A: 前序遍历，如果左子树为空右子树非空则插入空括号。

```cpp
class Solution {
public:
    string tree2str(TreeNode* root) {
        string ans = to_string(root->val);
        if (root->left) //left side check
            ans += "(" + tree2str(root->left) + ")";
        if (root->right) { //right side check
            if (!root->left) ans += "()"; //left side not present, but right side present
            ans += "(" + tree2str(root->right) + ")"; 
        }
        return ans;
    }
};
```

## [Minimum Distance Between BST Nodes](https://leetcode.com/problems/minimum-distance-between-bst-nodes)

A: 中序遍历得到有序结点，遍历寻找最小差值。

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func minDiffInBST(root *TreeNode) int {
    // declare the slice
    order := make([]int, 0, 100)

    // inorder traversal
    inorder(root, &order)

    // find the minimum difference
    ans := math.MaxInt
    for i := 1; i < len(order); i++ {
        if order[i] - order[i - 1] < ans {
            ans = order[i] - order[i - 1]
        }
    }
    return ans
}

func inorder(node *TreeNode, order *[]int) {
    if node != nil {
        inorder(node.Left, order)
        *order = append(*order, node.Val)
        inorder(node.Right, order)
    }
}
```

## [Average of Levels in Binary Tree](https://leetcode.com/problems/average-of-levels-in-binary-tree)

A: BFS。

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func averageOfLevels(root *TreeNode) []float64 {
    ans := []float64{}
    q := []*TreeNode{root}
    if root == nil {
        return ans
    }
    for len(q) > 0 {
        var sum float64
        size := len(q)
        for i := size; i > 0; i-- {
            cur := q[0]
            sum += float64(cur.Val)
            q = q[1:]
            if cur.Left != nil {
                q = append(q, cur.Left)
            }
            if cur.Right != nil {
                q = append(q, cur.Right)
            }
        }
        ans = append(ans, sum / float64(size))
    }
    return ans
}
```

## [N-ary Tree Level Order Traversal](https://leetcode.com/problems/n-ary-tree-level-order-traversal)

A: BFS。

```go
/**
 * Definition for a Node.
 * type Node struct {
 *     Val int
 *     Children []*Node
 * }
 */

func levelOrder(root *Node) [][]int {
    ans := [][]int{}
    q := []*Node{root}
    if root == nil {
        return ans
    }
    for len(q) > 0 {
        level := []int{}
        for i := len(q); i > 0; i-- {
            cur := q[0]
            q = q[1:]
            level = append(level, cur.Val)
            for _, c := range cur.Children {
                q = append(q, c)
            }
        }
        ans = append(ans, level)
    }
    return ans
}
```

## [Find Largest Value in Each Tree Row](https://leetcode.com/problems/find-largest-value-in-each-tree-row)

A: BFS。

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func largestValues(root *TreeNode) []int {
    ans := []int{}
    q := []*TreeNode{root}
    if root == nil {
        return ans
    }
    for len(q) > 0 {
        m := q[0].Val
        for i := len(q); i > 0; i-- {
            cur := q[0]
            if cur.Val > m {
                m = cur.Val
            }
            q = q[1:]
            if cur.Left != nil {
                q = append(q, cur.Left)
            }
            if cur.Right != nil {
                q = append(q, cur.Right)
            }
        }
        ans = append(ans, m)
    }
    return ans
}
```

## [Minimum Depth of Binary Tree](https://leetcode.com/problems/minimum-depth-of-binary-tree)

A: BFS，发现叶子节点立刻返回。

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func minDepth(root *TreeNode) int {
    if root == nil {
        return 0
    } 
    q := []*TreeNode{root}
    ans := 0
    for len(q) != 0 {
        ans++
        size := len(q)
        for i := 0; i < size; i++ {
            cur := q[0]
            q = q[1:]
            if cur.Left != nil {
                q = append(q, cur.Left)
            }
            if cur.Right != nil {
                q = append(q, cur.Right)
            }
            if cur.Right == nil && cur.Left == nil {
                return ans
            }
        }
    }
    return ans
}
```

## [Symmetric Tree](https://leetcode.com/problems/symmetric-tree)

A: 同时遍历两棵树（内侧和外侧），判断是否相等。

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func isSymmetric(root *TreeNode) bool {
    if root == nil {
        return true
    }
    if root.Left != nil && root.Right != nil {
        return isValid(root.Left, root.Right)
    } else if root.Left == nil && root.Right == nil {
        return true
    } else {
        return false
    }
}

func isValid(n1, n2 *TreeNode) bool {
    if n1 == nil && n2 == nil {
        return true
    }
    if n1 != nil && n2 != nil && n1.Val == n2.Val {
        return isValid(n1.Right, n2.Left) && isValid(n1.Left, n2.Right)
    }
    return false
}
```

## [Binary Tree Zigzag Level Order Traversal](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal)

A: BFS，每层反转。

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func zigzagLevelOrder(root *TreeNode) [][]int {
    ans := [][]int{}
    if root == nil {
        return nil
    }
    q := []*TreeNode{root}
    cnt := 1
    for len(q) > 0 {
        level := []int{}
        for i := len(q); i > 0; i-- {
            cur := q[0]
            q = q[1:]
            if cur.Left != nil {
                q = append(q, cur.Left)
            }
            if cur.Right != nil {
                q = append(q, cur.Right)
            }
            if cnt % 2 != 0 {
                // from left to right
                level = append(level, cur.Val)
            } else {
                // from right to left
                level = append(level, 0)
                copy(level[1:], level)
                level[0] = cur.Val
            }
        }
        ans = append(ans, level)
        cnt++
    }
    return ans
}
```

## [Count Complete Tree Nodes](https://leetcode.com/problems/count-complete-tree-nodes)

A: 简单做法可以直接遍历，计算节点个数，时间复杂度为O(n)。

效率更高可以使用分治思想，先计算左右子树的高度，如果相等，说明是满二叉树，节点个数为2^h-1。如果不相等，则利用相同方法，**分别计算左右子树中满二叉树的节点个数**，再加上根节点，即为总节点个数。

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */

func countNodes(root *TreeNode) int {
    if root == nil {
        return 0
    }
    leftDepth, rightDepth := 0, 0
    left, right := root.Left, root.Right
    for left != nil {
        left = left.Left
        leftDepth++
    }
    for right != nil {
        right = right.Right
        rightDepth++
    }
    if leftDepth == rightDepth {
        return (2 << leftDepth) - 1
    }
    return countNodes(root.Left) + countNodes(root.Right) + 1
}
```

## [Binary Tree Paths](https://leetcode.com/problems/binary-tree-paths)

A: DFS + 回溯。

```go
func binaryTreePaths(root *TreeNode) []string {
    res := make([]string, 0)
    var travel func(node *TreeNode, s string)
    travel = func(node *TreeNode, s string) {
        if node.Left == nil && node.Right == nil {
            v := s + strconv.Itoa(node.Val)
            res = append(res, v)
            return
        }
        s = s + strconv.Itoa(node.Val) + "->"
        if node.Left != nil {
            travel(node.Left, s)
        }
        if node.Right != nil {
            travel(node.Right, s)
        }
    }
    travel(root, "")
    return res
}
```

## [Sum of Left Leaves](https://leetcode.com/problems/sum-of-left-leaves)

A: 通过父节点、当前节点来判断是否为左叶子。

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func sumOfLeftLeaves(root *TreeNode) int {
    ans := 0
    dfs(root, &TreeNode{}, &ans)
    return ans
}

func dfs(root, pre *TreeNode, ans *int) {
    if root.Left == nil && root.Right == nil && pre.Left == root {
        *ans += root.Val
        return
    }
    if root.Left != nil {
        dfs(root.Left, root, ans)
    }
    if root.Right != nil {
        dfs(root.Right, root, ans)
    }
}
```

## [Maximum Binary Tree](https://leetcode.com/problems/maximum-binary-tree)

A: 寻找最大元素，确定前缀后缀，递归构造结点。

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    TreeNode* constructMaximumBinaryTree(vector<int>& nums) {
        int i = 0, j = nums.size();
        return build(nums, i, j);
    }

    TreeNode* build(vector<int>& nums, int i, int j) {
        if (i >= j) {
            return nullptr;
        }
        auto maxE = max_element(nums.begin() + i, nums.begin() + j);
        TreeNode *ans = new TreeNode(*maxE);
        ans->left = build(nums, i, maxE - nums.begin());
        ans->right = build(nums, maxE - nums.begin() + 1, j);
        return ans;
    }
};
```

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func constructMaximumBinaryTree(nums []int) *TreeNode {
    if len(nums) == 0 {
        return nil
    }
    maxIdx, maxNum := 0, 0
    for k, v := range nums {
        if v > maxNum {
            maxNum = v
            maxIdx = k
        }
    }
    root := &TreeNode{Val: maxNum}
    root.Left = constructMaximumBinaryTree(nums[:maxIdx])
    root.Right = constructMaximumBinaryTree(nums[maxIdx+1:])
    return root
}
```

## [Search in a Binary Search Tree](https://leetcode.com/problems/search-in-a-binary-search-tree)

A: 类似二分查找。

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func searchBST(root *TreeNode, val int) *TreeNode {
    if root == nil {
        return nil
    }
    if root.Val == val {
        return root
    } else if root.Val > val {
        return searchBST(root.Left, val)
    } else {
        return searchBST(root.Right, val)
    }
}
```

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    TreeNode* searchBST(TreeNode* root, int val) {
        if (!root) {
            return nullptr;
        }
        if (root->val == val) {
            return root;
        } else if (root->val < val) {
            return searchBST(root->right, val);
        } else {
            return searchBST(root->left, val);
        }
    }
};
```

## [Minimum Absolute Difference in BST](https://leetcode.com/problems/minimum-absolute-difference-in-bst)

A: 中序遍历后寻找最小差值。

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func getMinimumDifference(root *TreeNode) int {
    order := []int{}
    inorder(root, &order)
    ans := math.MaxInt
    for i := 1; i < len(order); i++ {
        if order[i] - order[i - 1] < ans {
            ans = order[i] - order[i - 1]
        }
    }
    return ans
}

func inorder(root *TreeNode, order *[]int) {
    if root == nil {
        return
    }
    inorder(root.Left, order)
    *order = append(*order, root.Val)
    inorder(root.Right, order)
}
```

## [Find Mode in Binary Search Tree](https://leetcode.com/problems/find-mode-in-binary-search-tree)

A: 中序遍历时保留前一个节点，判断是否相等，更新计数器。

```go
func findMode(root *TreeNode) []int {
    res := make([]int, 0)
    count := 1
    max := 1
    var prev *TreeNode
    var travel func(node *TreeNode) 
    travel = func(node *TreeNode) {
        if node == nil {
            return
        }
        travel(node.Left)
        if prev != nil && prev.Val == node.Val {
            count++
        } else {
            count = 1
        }
        if count >= max {
            if count > max && len(res) > 0 {
                res = []int{node.Val}
            } else {
                res = append(res, node.Val)
            }
            max = count
        }
        prev = node
        travel(node.Right)
    }
    travel(root)
    return res
}
```

## [Lowest Common Ancestor of a Binary Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree)

A: 在左右子树中分别查找p、q，如果左右子树都不为空，则当前节点为最近公共祖先，否则返回左右子树中不为空的节点。

```go
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
    // check
    if root == nil {
        return root
    }
    // 相等 直接返回root节点即可
    if root == p || root == q {
        return root
    }
    // Divide
    left := lowestCommonAncestor(root.Left, p, q)
    right := lowestCommonAncestor(root.Right, p, q)

    // Conquer
    // 左右两边都不为空，则根节点为祖先
    if left != nil && right != nil {
        return root
    }
    if left != nil {
        return left
    }
    if right != nil {
        return right
    }
    return nil
}
```

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if (!root) {
            return nullptr;
        }

        if (root == p || root == q) {
            return root;
        }

        auto left = lowestCommonAncestor(root->left, p, q);
        auto right = lowestCommonAncestor(root->right, p, q);
        
        if (left && right) {
            return root;
        } else if (left) {
            return left;
        } else {
            return right;
        }
    }
};
```

## [Construct Quad Tree](https://leetcode.com/problems/construct-quad-tree)

A: 遇到全0或全1则返回对应节点，否则递归构造子节点。

```go
/**
 * Definition for a QuadTree node.
 * type Node struct {
 *     Val bool
 *     IsLeaf bool
 *     TopLeft *Node
 *     TopRight *Node
 *     BottomLeft *Node
 *     BottomRight *Node
 * }
 */

func construct(grid [][]int) *Node {
    return dfs(grid)
}

func dfs(grid [][]int) *Node {
    head := &Node{}
    prev := grid[0][0]
    for i := 0; i < len(grid); i++ {
        for j := 0; j < len(grid); j++ {
            if grid[i][j] != prev {
                // split by 4
                mid := len(grid) / 2
                topLeft := make([][]int, 0, mid)
                topRight := make([][]int, 0, mid)
                bottomLeft := make([][]int, 0, mid)
                 bottomRight := make([][]int, 0, mid)
                for k := 0; k < mid; k++ {
                    topLeft = append(topLeft, grid[k][:mid])
                    topRight = append(topRight, grid[k][mid:])
                    bottomLeft = append(bottomLeft, grid[mid+k][:mid])
                    bottomRight = append(bottomRight, grid[mid+k][mid:])
                }
                head.TopLeft = dfs(topLeft)
                head.TopRight = dfs(topRight)
                head.BottomLeft = dfs(bottomLeft)
                head.BottomRight = dfs(bottomRight)
                return head
            }
        }
    }
    if prev == 1 {
        head.Val = true
    } else {
        head.Val = false
    }
    head.IsLeaf = true
    return head
}
```

## [Find Duplicate Subtrees](https://leetcode.com/problems/find-duplicate-subtrees)

A: 通过中序遍历构造字符串，使用map记录每个字符串出现的次数，出现两次则为重复子树。

```go
func findDuplicateSubtrees(root *TreeNode) []*TreeNode {
	hashAll := map[string]int{}
	duplicate := []*TreeNode{}
	dfs(root, hashAll, &duplicate)
	return duplicate
}

func dfs(node *TreeNode, hashAll map[string]int, duplicate *[]*TreeNode) string {
	if node == nil {
		return "nil"
	}
    lString := dfs(node.Left, hashAll, duplicate)
    rString := dfs(node.Right, hashAll, duplicate)
    buildString := fmt.Sprintf("(%s)(%v)(%s)", lString, node.Val, rString)
	hashAll[buildString]++
	if hashAll[buildString] == 2 {
		*duplicate = append(*duplicate, node)
	}
    return buildString
}
```

A: 记录后序遍历序列。

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    unordered_map<string, pair<TreeNode*, int>> map;
    vector<TreeNode*> ans;
    vector<TreeNode*> findDuplicateSubtrees(TreeNode* root) {
        dfs(root);
        for (auto &p : map) {
            if (p.second.second > 1) {
                ans.push_back(p.second.first);
            }
        }
        return ans;
    }

    string dfs(TreeNode* root) {
        if (!root) {
            return "#";
        }

        string lstr = dfs(root->left);
        string rstr = dfs(root->right);
        string ans = lstr + "," + rstr + "," + to_string(root->val);
        if (map.count(ans)) {
            map[ans].second++;
        } else {
            map[ans] = {root, 1};
        }
        return ans;
    }
};
```

## [Binary Tree Cameras](https://leetcode.com/problems/binary-tree-cameras)

A: 贪心，优先安装在叶子节点的父节点上，这意味着需要**后序遍历**，对于每个节点，有三种状态：

- a: 该节点安装了摄像头
- b: 该节点的父节点安装了摄像头
- c: 该节点的子节点安装了摄像头

```go
const inf = math.MaxInt64 / 2

func minCameraCover(root *TreeNode) int {
    var dfs func(*TreeNode) (a, b, c int)
    dfs = func(node *TreeNode) (a, b, c int) {
        if node == nil {
            // 空节点，不需要安装摄像头
            return inf, 0, 0
        }
        lefta, leftb, leftc := dfs(node.Left)
        righta, rightb, rightc := dfs(node.Right)
        // a, b, c分别表示当前节点安装摄像头、父节点安装摄像头、子节点安装摄像头的最小值
        a = leftc + rightc + 1 // 当前节点安装摄像头，则左右子节点不需要安装摄像头
        b = min(a, min(lefta+rightb, righta+leftb)) // 父节点安装摄像头，则左右子节点可以安装也可以不安装
        c = min(a, leftb+rightb) // 子节点安装摄像头，则左右子节点必须安装摄像头
        return
    }
    _, ans, _ := dfs(root)
    return ans
}

func min(a, b int) int {
    if a <= b {
        return a
    }
    return b
}
```

```cpp
class Solution {
private:
    int result;
    int traversal(TreeNode* cur) {
        if (cur == NULL) return 2;
        int left = traversal(cur->left);    // 左
        int right = traversal(cur->right);  // 右
        if (left == 2 && right == 2) return 0;
        else if (left == 0 || right == 0) {
            result++;
            return 1;
        } else return 2;
    }
public:
    int minCameraCover(TreeNode* root) {
        result = 0;
        if (traversal(root) == 0) { // root 无覆盖
            result++;
        }
        return result;
    }
};
```

## [Kth Largest Sum in A Binary Tree](https://leetcode.com/problems/kth-largest-sum-in-a-binary-tree)

A: BFS，排序得到答案。

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func kthLargestLevelSum(root *TreeNode, k int) int64 {
    levels := []int64{}
    q := []*TreeNode{root}
    for len(q) != 0 {
        var curSum int64 = 0
        for i := len(q) - 1; i >= 0; i-- {
            cur := q[0]
            q = q[1:]
            curSum += int64(cur.Val)
            if cur.Left != nil {
                q = append(q, cur.Left)
            }
            if cur.Right != nil {
                q = append(q, cur.Right)
            }
        }
        levels = append(levels, curSum)
    }
    if len(levels) < k {
        return -1
    }
    sort.Slice(levels, func(i, j int) bool {
        return levels[i] < levels[j]
    })
    return levels[len(levels) - k]
}
```

## [Count Number of Possible Root Nodes](https://leetcode.com/problems/count-number-of-possible-root-nodes)

A: re-rooting，将每个节点作为根节点，统计每个节点的子树中有多少个节点满足条件。

[详解](https://leetcode.com/problems/count-number-of-possible-root-nodes/solutions/3256065/re-rooting-o-n-explained/)

```cpp
class Solution {
public:
    unordered_map<int,vector<int>> adj;
    map<vector<int>,int> g;
    int ans=0;
    int k;
    void setpar(int node,int parent,vector<int> &par){
        par[node]=parent;
        for(auto child : adj[node]){
            if(child!=parent){
                setpar(child,node,par);
            }
        }
    }

    void dfs(int node,int par,int guess){
        if(g[{node,par}]){
            guess++;
        }
        if(g[{par,node}]){
            guess--;
        }
        for(auto child : adj[node]){
            if(child!=par){
                dfs(child,node,guess);
            }
        }
        if(guess>=k){
            ans++;
        }
    }

    int rootCount(vector<vector<int>>& edges, vector<vector<int>>& guesses, int k) {
        this->k=k;
        int n=edges.size()+1;

        for(int i=0 ; i<edges.size() ; i++){             // filling adjacent of each node
            adj[edges[i][0]].push_back(edges[i][1]);
            adj[edges[i][1]].push_back(edges[i][0]);
        }

        for(int i=0 ; i<guesses.size() ; i++){          //mapping guesses 
            g[guesses[i]]=1;
        }

        vector<int> par(edges.size()+1,0);

        setpar(0,-1,par);                                  // first dfs - 0 node as root
        int guess=0;
        for(int i=1;i<n ; i++){
            if(g[{par[i],i}]==1){
                guess++;
            }
        }

        
        if(guess>=k){
            ans++;
        }

        for(auto child : adj[0]){                        // second dfs - taking nodes except 0 as root
            dfs(child,0,guess);
        }

        return ans;

    }
};
```

## [Convert Sorted Array to Binary Search Tree](https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree)

A: 快慢指针找中间结点，然后分段构造左右子树。

```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func sortedListToBST(head *ListNode) *TreeNode {
    return buildTree(head, nil)
}

func buildTree(begin, end *ListNode) *TreeNode {
    fmt.Println(begin, end)
    if begin == end {
        return nil
    }
    fast, slow := begin, begin
    for fast != end && fast.Next != end {
        slow = slow.Next
        fast = fast.Next.Next
    }
    root := &TreeNode{Val: slow.Val}
    root.Left = buildTree(begin, slow)
    root.Right = buildTree(slow.Next, end)
    return root
}
```

## [Check Completeness of a Binary Tree](https://leetcode.com/problems/check-completeness-of-a-binary-tree)

A: BFS，当处理了空节点后，就不应该再有非空节点。

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func isCompleteTree(root *TreeNode) bool {
    q := []*TreeNode{root}
    flag := false
    for len(q) != 0 {
        for i := len(q); i > 0; i-- {
            cur := q[0]
            q = q[1:]
            if cur != nil {
                if flag {
                    return false
                }
                q = append(q, cur.Left)
                q = append(q, cur.Right)
            } else {
                flag = true
            }
        }
    }
    return true
}
```

## [Balance a Binary Search Tree](https://leetcode.com/problems/balance-a-binary-search-tree)

A: 中序遍历得到有序数组，然后递归构造平衡二叉树。

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func balanceBST(root *TreeNode) *TreeNode {
   // 二叉搜索树中序遍历得到有序数组
	nums := []int{}
   // 中序递归遍历二叉树
	var travel func(node *TreeNode)
	travel = func(node *TreeNode) {
		if node == nil {
			return
		}
		travel(node.Left)
		nums = append(nums, node.Val)
		travel(node.Right)
	}
	// 二分法保证左右子树高度差不超过一（题目要求返回的仍是二叉搜索树）
	var buildTree func(nums []int, left, right int) *TreeNode
	buildTree = func(nums []int, left, right int) *TreeNode {
		if left > right {
			return nil
		}
		mid := left + (right-left) >> 1
		root := &TreeNode{Val: nums[mid]}
		root.Left = buildTree(nums, left, mid-1)
		root.Right = buildTree(nums, mid+1, right)
		return root
	}
	travel(root)
	return buildTree(nums, 0, len(nums)-1)
}
```

## [Longest ZigZag Path in a Binary Tree](https://leetcode.com/problems/longest-zigzag-path-in-a-binary-tree)

A: 递归，返回三个值，分别是以当前节点为根的最长ZigZag路径，以当前节点为根的左子树的最长ZigZag路径，以当前节点为根的右子树的最长ZigZag路径。注意空结点返回-1。

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    int longestZigZag(TreeNode* root) {
        return dfs(root)[0];
    }

    vector<int> dfs(TreeNode* root) {
        // ans: ans, left, right
        if (root == nullptr) {
            return {-1, -1, -1};
        }
        auto left = dfs(root->left);
        auto right = dfs(root->right);
        return {
            max({1 + max(left[2], right[1]), left[0], right[0]}),
            left[2] + 1,
            right[1] + 1
        };
    }
};
```

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func longestZigZag(root *TreeNode) int {
    ans, l, r := dfs(root)
    fmt.Println(ans, l, r)
    return ans
}

func dfs(root *TreeNode) (ans, left, right int) {
    if root == nil {
        ans, left, right = -1, -1, -1
        fmt.Println(root, ans, left, right)
        return 
    }
    lans, _, lright := dfs(root.Left)
    rans, rleft, _ := dfs(root.Right)
    ans = max(max(lright, rleft) + 1, lans, rans)
    left = lright + 1
    right = rleft + 1
    return 
}

func max(nums ...int) int {
    ans := -1
    for _, n := range nums {
        if n > ans {
            ans = n
        }
    }
    return ans
}
```

## [Maximum Width of Binary Tree](https://leetcode.com/problems/maximum-width-of-binary-tree)

A: 完全二叉树的**子节点编号**为2i+1和2i+2，所以可以用层序遍历的方法，每次遍历一层，记录每个节点的编号，然后计算最大宽度。

```cpp
class Solution {
public:
    int widthOfBinaryTree(TreeNode* root) {
        if(root == NULL)
            return 0;
        
        int res = 1;
        queue<pair<TreeNode*, int>> q;
        
        // I am using intialising list
        q.push({root, 0});      // also can use make_pair
        
        while(!q.empty())
        {
            int cnt = q.size();
            // start is the index of root node for first level
            int start = q.front().second;
            int end = q.back().second;
            
            res = max(res,end-start + 1);
            
            for(int i = 0; i <cnt; ++i)
            {
                pair<TreeNode*, int> p = q.front();
                // we will use it while inserting it children
                // left child will be 2 * idx + 1;
                // right chils will be 2 * idx + 2;
                int idx = p.second - start;
                
                q.pop();
                
                // if  left child exist
                if(p.first->left != NULL)
                    q.push({p.first->left, (long long)2 * idx + 1});
                
                // if right child exist
                if(p.first->right != NULL)
                    q.push({p.first->right, (long long) 2 * idx + 2});
            }
        }
        
        return res;
        
    }
};
```

## [Flatten Binary Tree to Linked List](https://leetcode.com/problems/flatten-binary-tree-to-linked-list)

A: 后序遍历，先将左右子树拉平，然后将左子树接到右子树上，最后将右子树接到左子树的最右边。

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    void flatten(TreeNode* root) {
        if (!root) {
            return;
        }
        flatten(root->left);
        flatten(root->right);
        auto left = root->left;
        auto right = root->right;
        root->right = left;
        root->left = nullptr;
        auto cur = root;
        while (cur->right != nullptr) {
            cur = cur->right;
        }
        cur->right = right;
    }
};
```

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func flatten(root *TreeNode)  {
    if root == nil {
        return
    }
    flatten(root.Left)
    flatten(root.Right)
    l := root.Left
    r := root.Right
    root.Left = nil
    root.Right = l
    cur := root
    for cur.Right != nil {
        cur = cur.Right
    }
    cur.Right = r
}
```

## [Time Needed to Inform All Employees](https://leetcode.com/problems/time-needed-to-inform-all-employees)

A: 从叶子节点开始向上遍历，记录每个节点的最大路径和，最后取最大值。

```go
func numOfMinutes(n int, headID int, manager []int, informTime []int) int {
    totalTime := [100000]int{}
    
    /*
    traverse from leaf to head and track the max path-sum, memo results
    */
    maxTime := 0
    for i := 0; i < n; i++ {
        if totalTime[i] > 0 {
            continue
        }
        
        totTime := 0
        curr := i
        for curr != headID {
            mgr := manager[curr]
            if totalTime[mgr] > 0 {
                totTime += (totalTime[mgr] + informTime[mgr])
                break
            }
            totTime += informTime[mgr]
            curr = mgr
        }
        
        totalTime[i] = totTime
        
        maxTime = max(maxTime, totTime)
    } 
    
    return maxTime
}

func max(v1, v2 int) int {
    if v1 > v2 {
        return v1
    }
    return v2
}
```

## [二叉树的下一个节点](https://www.nowcoder.com/practice/9023a0c988684a53960365b889ceaf5e?tpId=265&tags=&title=&difficulty=0&judgeStatus=0&rp=1&sourceUrl=%2Fexam%2Foj%2Fta%3Fpage%3D1%26tpId%3D13%26type%3D265)

A: 分类讨论，如果当前节点有右子树，则下一个节点为右子树的最左节点；如果当前节点没有右子树，**则向上找，直到找到一个节点是其父节点的左子树，则下一个节点为该父节点**。

```go
func GetNext(pNode *TreeLinkNode) *TreeLinkNode {
    if pNode == nil {
        return nil
    }
    cur := pNode
    var nxt *TreeLinkNode
    if cur.Right != nil {
        nxt = cur.Right
        for nxt.Left != nil {
            nxt = nxt.Left
        }
    } else if cur.Next != nil {
        nxt = cur.Next
        for nxt != nil && nxt.Right == cur {
            cur = nxt
            nxt = nxt.Next
        }
    }
    return nxt
}
```

## [二叉搜索树的后序遍历序列](https://www.nowcoder.com/practice/a861533d45854474ac791d90e447bafd?tpId=265&tags=&title=&difficulty=0&judgeStatus=0&rp=1&sourceUrl=%2Fexam%2Foj%2Fta%3FtpId%3D13)

A: 抓住BST后序遍历的大小关系，将序列分为左子树、右子树和根节点，然后递归判断左右子树是否为BST。

```go
func VerifySquenceOfBST( sequence []int ) bool {
    if len(sequence) == 0 {
        return false
    }
    if len(sequence) == 1 {
        return true
    }
    root := sequence[len(sequence)-1]
    leftEnd, rightStart := -1, -1
    for i := 0; i < len(sequence) - 1; i++ {
        if sequence[i] >= root {
            leftEnd = i
            break
        }
    }
    for i := len(sequence) - 2; i >= 0; i-- {
        if sequence[i] < root {
            rightStart = i + 1
            break
        }
    }
    if leftEnd != rightStart && leftEnd != -1 && rightStart != -1 {
        return false
    }
    if leftEnd == 0 || rightStart == len(sequence) - 1 {
        return VerifySquenceOfBST(sequence[:len(sequence)-1])
    }
    return VerifySquenceOfBST(sequence[:leftEnd]) && VerifySquenceOfBST(sequence[leftEnd:len(sequence)-1])
}
```
