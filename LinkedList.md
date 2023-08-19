# Linked List

## [Reverse Linked List](https://leetcode.com/problems/reverse-linked-list)

A: 循环，当前节点（head)、前一个节点、后一个节点依次交换赋值。

```cpp
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode *nextNode, *prevNode = nullptr;
        while (head) {
            nextNode = head->next;
            head->next = prevNode;
            prevNode = head;
            head = nextNode;
        }
        return prevNode;
    }
};
```

```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func reverseList(head *ListNode) *ListNode {
    if head == nil {
        return head
    }
    dummy := &ListNode{}
    pre, cur, next := dummy, head, head.Next
    for next != nil {
        cur.Next = pre
        pre = cur
        cur = next
        next = next.Next
    }
    cur.Next = pre
    head.Next = nil
    return cur
}
```

A: 递归，当前节点（head)、前一个节点、后一个节点依次交换赋值。

```cpp
class Solution {
public:
    ListNode* reverseList(ListNode *head, ListNode *nextNode = nullptr, ListNode *prevNode = nullptr) {
        // 给定函数参数时完成赋值
        return head ? reverseList(head->next, (head->next = prevNode, nextNode), head) : prevNode;
    }
};
```

## [Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists)

A: 虚拟头节点，两列表非空，则依次比较；有一为空，则连接剩下非空列表。

```cpp
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        ListNode *dummy, *temp;
        dummy = new ListNode();
        temp = dummy;
        
        // when both list1 and list2 isn't empty
        while(list1 && list2){
            if(list1->val < list2->val){
                temp->next = list1;
                list1 = list1->next;
            }
            else{
                temp->next = list2;
                list2 = list2->next;   
            }
            temp = temp->next;
        }
        
        // we reached at the end of one of the list
        if(list1) temp->next = list1;
        if(list2) temp->next = list2;
        
        return dummy->next;
    }
};
```

```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func mergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
    dummy := &ListNode{}
    cur := dummy
    for list1 != nil && list2 != nil {
        if list1.Val < list2.Val {
            cur.Next = list1
            list1 = list1.Next
        } else {
            cur.Next = list2
            list2 = list2.Next
        }
        cur = cur.Next
    }
    if list1 != nil {
        cur.Next = list1
    }
    if list2 != nil {
        cur.Next = list2
    }
    return dummy.Next
}
```

## [Reorder List](https://leetcode.com/problems/reorder-list)

A: 用栈存放逆序列表，以size/2为遍历终止条件。

```cpp
class Solution {
public:
    void reorderList(ListNode* head) {
        if ((!head) || (!head->next) || (!head->next->next)) return; // Edge cases
        
        stack<ListNode*> my_stack;
        ListNode* ptr = head;
        int size = 0;
        while (ptr != NULL) // Put all nodes in stack
        {
            my_stack.push(ptr);
            size++;
            ptr = ptr->next;
        }
        
        ListNode* pptr = head;
        for (int j=0; j<size/2; j++) // Between every two nodes insert the one in the top of the stack
        {
            ListNode *element = my_stack.top();
            my_stack.pop();
            element->next = pptr->next;
            pptr->next = element;
            pptr = pptr->next->next;
        }
        pptr->next = NULL;
    }
};
```

A: 快慢指针找到切分点，逆序后侧链表，合并两部分链表。

```cpp
class Solution {
public:
    void reorderList(ListNode* head) {
        if (head->next == NULL) {
            return;
        }
        
        ListNode* prev = NULL;
        ListNode* slow = head;
        ListNode* fast = head;
        
        while (fast != NULL && fast->next != NULL) {
            prev = slow;
            slow = slow->next;
            fast = fast->next->next;
        }
        
        prev->next = NULL;
        
        ListNode* l1 = head;
        ListNode* l2 = reverse(slow);
        
        merge(l1, l2);
    }
private:
    ListNode* reverse(ListNode* head) {
        ListNode* prev = NULL;
        ListNode* curr = head;
        ListNode* next = curr->next;
        
        while (curr != NULL) {
            next = curr->next;
            curr->next = prev;
            prev = curr;
            curr = next;
        }
        
        return prev;
    }
    void merge(ListNode* l1, ListNode* l2) {
        while (l1 != NULL) {
            ListNode* p1 = l1->next;
            ListNode* p2 = l2->next;
            
            l1->next = l2;
            if (p1 == NULL) {
                break;
            }
            l2->next = p1;
            
            l1 = p1;
            l2 = p2;
        }
    }
};
```

```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func reorderList(head *ListNode)  {
    slow, fast := head, head
    for fast != nil && fast.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
    }
    pre, cur := slow, slow.Next
    pre.Next = nil
    var next *ListNode
    if cur != nil {
        next = cur.Next
    }
    for cur != nil {
        cur.Next = pre
        pre = cur
        cur = next
        if next == nil {
            break
        }
        next = next.Next
    }
    left, right := head.Next, pre
    cur = head
    for left != nil || right != nil {
        if cur.Next == cur {
            cur.Next = nil
            break
        }
        if right != nil {
            cur.Next = right
            right = right.Next
            cur = cur.Next
        }
        if left != nil {
            cur.Next = left
            left = left.Next
            cur = cur.Next
        }
    }
}
```

## [Remove Nth Node From End of List](https://leetcode.com/problems/remove-nth-node-from-end-of-list)

A: 快慢指针，快指针比慢指针初始快n位；考虑删除头结点的边界情况。

```cpp
ListNode* removeNthFromEnd(ListNode* head, int n) {
    ListNode *fast = head, *slow = head;
    while(n--) fast = fast -> next;      // iterate first n nodes using fast
    if(!fast) return head -> next;       // if fast is already null, it means we have to delete head itself. So, just return next of head
    while(fast -> next)                  // iterate till fast reaches the last node of list
        fast = fast -> next, slow = slow -> next;            
    slow -> next = slow -> next -> next; // remove the nth node from last
    return head;
}
```

```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func removeNthFromEnd(head *ListNode, n int) *ListNode {
    dummy := &ListNode{Next: head}
    slow, fast := dummy, dummy
    for i := 0; i < n; i++ {
        fast = fast.Next
    }
    var pre, next *ListNode
    for fast != nil {
        fast = fast.Next
        pre = slow
        slow = slow.Next
        next = slow.Next
    }
    pre.Next = next
    return dummy.Next
}
```

## [Linked List Cycle](https://leetcode.com/problems/linked-list-cycle)

A: 快慢指针，快指针比慢指针快一倍，两者相遇则有环。

```cpp
class Solution {
public:
    bool hasCycle(ListNode *head) {
       ListNode *fast = head, *slow = head;
       while(fast && slow) {
           if (fast->next) fast = fast->next->next;
           else return false;
           slow = slow->next;
           if (slow == fast) return true;
       }
       return false;
    }
};
```

```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func hasCycle(head *ListNode) bool {
    slow, fast := head, head
    for fast != nil && fast.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
        if slow == fast {
            return true
        }
    }
    return false
}
```

## [Merge K Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists)

A: 每次取两个列表，两两合并。

```cpp
class Solution {
public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        int n = lists.size();
        if (n == 0) {
            return NULL;
        }
        
        while (n > 1) {
            for (int i = 0; i < n / 2; i++) {
                lists[i] = mergeTwoLists(lists[i], lists[n - i - 1]);
            }
            n = (n + 1) / 2;
        }
        
        return lists.front();
    }
private:
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        if (list1 == nullptr && list2 == nullptr) {
            return nullptr;
        }
        if (list1 == nullptr) {
            return list2;
        }
        if (list2 == nullptr) {
            return list1;
        }
        
        ListNode* head = nullptr;
        if (list1->val <= list2->val) {
            head = list1;
            list1 = list1->next;
        } else {
            head = list2;
            list2 = list2->next;
        }
        ListNode* curr = head;
        
        while (list1 != nullptr && list2 != nullptr) {
            if (list1->val <= list2->val) {
                curr->next = list1;
                list1 = list1->next;
            } else {
                curr->next = list2;
                list2 = list2->next;
            }
            curr = curr->next;
        }
        
        if (list1 == nullptr) {
            curr->next = list2;
        } else {
            curr->next = list1;
        }
        
        return head;
    }
};
```

```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func mergeKLists(lists []*ListNode) *ListNode {
    if len(lists) == 0 {
        return nil
    }
    for sz := 1; sz < len(lists); sz *= 2 {
        for i := 0; i < len(lists) - sz; i += 2 * sz {
            lists[i] = mergeLists(lists[i], lists[i+sz])
        }
    }
    return lists[0]
}

func mergeLists(list1, list2 *ListNode) *ListNode {
    if list1 == nil {
        return list2
    } else if list2 == nil {
        return list1
    }
    var head *ListNode
    if list1.Val < list2.Val {
        head = list1
        list1 = list1.Next
    } else {
        head = list2
        list2 = list2.Next
    }
    cur := head
    for list1 != nil && list2 != nil {
        if list1.Val < list2.Val {
            cur.Next = list1
            list1 = list1.Next
        } else {
            cur.Next = list2
            list2 = list2.Next
        }
        cur = cur.Next
    }
    if list1 != nil {
        cur.Next = list1
    }
    if list2 != nil {
        cur.Next = list2
    }
    return head
}
```

## [Copy List with Random Pointer](https://leetcode.com/problems/copy-list-with-random-pointer)

A: 用`map`存储结点的对应关系。

```cpp
class Solution {
public:
    Node* copyRandomList(Node* head) {
        map<Node*, Node*> m;
        int i=0;
        Node* ptr = head;
        while (ptr) {
            m[ptr] =new Node(ptr->val);
            ptr = ptr->next;
        }
        ptr = head;
        while (ptr) {
            m[ptr]->next = m[ptr->next];
            m[ptr]->random = m[ptr->random];
            ptr = ptr->next;
        }
        return m[head];
    }
};
```

## [Add Two Numbers](https://leetcode.com/problems/add-two-numbers)

A: 注意同时遍历多个链表的代码组织方式。

```cpp
ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
    int c = 0;
    ListNode newHead(0);
    ListNode *t = &newHead;
    while(c || l1 || l2) {
        c += (l1? l1->val : 0) + (l2? l2->val : 0);
        t->next = new ListNode(c%10);
        t = t->next;
        c /= 10;
        if(l1) l1 = l1->next;
        if(l2) l2 = l2->next;
    }
    return newHead.next;
}
```

## [Find The Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number)

A: 数组元素的索引指向另一个数组元素，问题转换为寻找链表环的入口，使用[yd Cycle Detection Algorithm](https://zh.wikipedia.org/wiki/Floyd%E5%88%A4%E5%9C%88%E7%AE%97%E6%B3%95)。

```cpp
class Solution {
public:
    int findDuplicate(vector<int>& nums) {
        int slow = nums[0];
        int fast = nums[nums[0]];
        
        while (slow != fast) {
            slow = nums[slow];
            fast = nums[nums[fast]];
        }
        
        slow = 0;
        while (slow != fast) {
            slow = nums[slow];
            fast = nums[fast];
        }
        return slow;
    }
};
```

## [Reverse Nodes In K Group](https://leetcode.com/problems/reverse-nodes-in-k-group)

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        ListNode* dummy = new ListNode(0);
        dummy->next = head;
        ListNode* before = dummy;
        ListNode* after = head;
        ListNode* curr = nullptr;
        ListNode* prev = nullptr;
        ListNode* nxt = nullptr;
        while(true){
            ListNode* cursor = after;
            for(int i = 0; i < k; i++){
                if(cursor == nullptr) return dummy->next; // 不足k个元素
                cursor = cursor->next;
            }
            curr = after;
            prev = before;
            for(int i = 0; i < k; i++){
                nxt = curr->next;
                curr->next = prev;
                prev = curr;
                curr = nxt;
            }
            after->next = curr;
            before->next = prev;
            before = after;
            after = curr;
        }
    }        
};
```

## [Swap Nodes In Pairs](https://leetcode.com/problems/swap-nodes-in-pairs)

A: 循环时注意指针移动时的指向位置。

```cpp
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        // if head is NULL OR just having a single node, then no need to change anything 
        if(head == NULL || head -> next == NULL) {
            return head;
        }
            
        ListNode* temp; // temporary pointer to store head -> next
        temp = head->next; // give temp what he want
        
        head->next = swapPairs(head->next->next); // changing links
        temp->next = head; // put temp -> next to head
        
        return temp; // now after changing links, temp act as our head
    }
};
```

```cpp
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        if (head == nullptr ||
            head->next == nullptr) {
            return head;
        }
        
        ListNode * prev = nullptr;
        ListNode * curr = head;
        ListNode * nxt  = head->next;
        
        head = head->next;
        
        while (curr && nxt) {
            if (prev)
                prev->next = nxt;
            curr->next = nxt->next;
            nxt->next = curr;
            prev = curr;
            curr = curr->next;
            if (curr)
                nxt = curr->next;
        }
        
        return head;
    }
};
```

```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func swapPairs(head *ListNode) *ListNode {
    if head == nil {
        return nil
    }
    dummy := &ListNode{Next: head}
    pre, cur, next := dummy, head, head.Next
    for cur != nil && next != nil {
        pre.Next = next
        cur.Next = next.Next
        next.Next = cur
        pre = cur
        cur = pre.Next
        if cur != nil {
            next = cur.Next
        }
    }
    return dummy.Next
}
```

## [Sort List](https://leetcode.com/problems/sort-list)

A: 归并排序，快慢指针得到中间点，切分，保证两边序列有序，然后合并。

```cpp
class Solution {
public:
    ListNode* sortList(ListNode* head) {
        //If List Contain a Single or 0 Node
        if(head == NULL || head ->next == NULL)
            return head;
        
        
        ListNode *temp = NULL;
        ListNode *slow = head;
        ListNode *fast = head;
        
        // 2 pointer appraoach / turtle-hare Algorithm (Finding the middle element)
        while(fast !=  NULL && fast -> next != NULL)
        {
            temp = slow;
            slow = slow->next;          //slow increment by 1
            fast = fast ->next ->next;  //fast incremented by 2
            
        }   
        temp -> next = NULL;            //end of first left half
        
        ListNode* l1 = sortList(head);    //left half recursive call
        ListNode* l2 = sortList(slow);    //right half recursive call
        
        return mergelist(l1, l2);         //mergelist Function call
            
    }
    
    //MergeSort Function O(n*logn)
    ListNode* mergelist(ListNode *l1, ListNode *l2)
    {
        ListNode *ptr = new ListNode(0);
        ListNode *curr = ptr;
        
        while(l1 != NULL && l2 != NULL)
        {
            if(l1->val <= l2->val)
            {
                curr -> next = l1;
                l1 = l1 -> next;
            }
            else
            {
                curr -> next = l2;
                l2 = l2 -> next;
            }
        
        curr = curr ->next;
        
        }
        
        //for unqual length linked list
        
        if(l1 != NULL)
        {
            curr -> next = l1;
            l1 = l1->next;
        }
        
        if(l2 != NULL)
        {
            curr -> next = l2;
            l2 = l2 ->next;
        }
        
        return ptr->next;
    }
};
```

## [Partition List](https://leetcode.com/problems/partition-list)

A: 维护两个部分对应的两个子list，最后拼接。

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* partition(ListNode* head, int x) {
        if (!head) return head;
        ListNode* dummy1 = new ListNode();
        ListNode* dummy2 = new ListNode();
        ListNode *cur = head, *cur1 = dummy1, *cur2 = dummy2;
        while (cur) {
            if (cur->val < x) {
                cur1->next = cur;
                cur1 = cur1->next;
            } else {
                cur2->next = cur;
                cur2 = cur2->next;
            }
            cur = cur->next;
        }
        cur1->next = dummy2->next;
        cur2->next = nullptr;
        return dummy1->next;
    }
};
```

```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func partition(head *ListNode, x int) *ListNode {
    if head == nil {
        return head
    }
    dummy1, dummy2 := new(ListNode), new(ListNode)
    cur := head
    var cur1, cur2 *ListNode = dummy1, dummy2
    for cur != nil {
        if cur.Val < x {
            cur1.Next = cur
            cur1 = cur1.Next
        } else {
            cur2.Next = cur
            cur2 = cur2.Next
        }
        cur = cur.Next
    }
    cur1.Next = dummy2.Next
    cur2.Next = nil
    return dummy1.Next
}
```

## [Design Linked List](https://leetcode.com/problems/design-linked-list)

A: 双向链表，虚拟头尾节点。

```cpp
class node {
public:
    int val = 0;
    node *next = nullptr;
    node *pre = nullptr;
};

class MyLinkedList {
private:
    node* dummy = new node();
    node* tail = new node();
    int len = 0;
public:
    MyLinkedList() {
        dummy->next = tail;
        tail->pre = dummy;
    }
    
    int get(int index) {
        if (index >= len) {
            return -1;
        }
        node *cur = dummy->next;
        for (int i = 0; i < index; i ++) {
            cur = cur->next;
        }
        return cur->val;
    }
    
    void addAtHead(int val) {
        node* tmp = new node();
        tmp->val = val;
        tmp->pre = dummy;
        dummy->next->pre = tmp;
        tmp->next = dummy->next;
        dummy->next = tmp;
        len++;
    }
    
    void addAtTail(int val) {
        node *tmp = new node();
        tmp->val = val;
        tmp->next = tail;
        tmp->pre = tail->pre;
        tail->pre->next = tmp;
        tail->pre = tmp;
        len++;
    }
    
    void addAtIndex(int index, int val) {
        if (index > len || index < 0) {
            return;
        }
        node *n = new node();
        n->val = val;
        node *cur = dummy;
        for (int i = -1; i < index; i++) {
            cur = cur->next;
        }
        n->next = cur;
        n->pre = cur->pre;
        cur->pre->next = n;
        cur->pre = n;
        len++;
    }
    
    void deleteAtIndex(int index) {
        if (index >= len || index < 0) {
            return;
        }
        node *cur = dummy;
        for (int i = -1; i < index; i++) {
            cur = cur->next;
        }
        cur->pre->next = cur->next;
        cur->next->pre = cur->pre;
        len--;
    }
};

/**
 * Your MyLinkedList object will be instantiated and called as such:
 * MyLinkedList* obj = new MyLinkedList();
 * int param_1 = obj->get(index);
 * obj->addAtHead(val);
 * obj->addAtTail(val);
 * obj->addAtIndex(index,val);
 * obj->deleteAtIndex(index);
 */
```

```go
type MyLinkedList struct {
    Head *Node
    Tail *Node
    cnt int
}

type Node struct {
    Val int
    Next *Node
    Pre *Node
}

func Constructor() MyLinkedList {
    head := &Node{}
    tail := &Node{Pre: head}
    head.Next = tail
    return MyLinkedList{Head: head, Tail: tail}
}


func (this *MyLinkedList) Get(index int) int {
    if !this.isValid(index) {
        return -1
    }
    cur := this.Head.Next
    for i := 0; i < index; i++ {
        cur = cur.Next
    }
    return cur.Val
}

func (this *MyLinkedList) AddAtHead(val int)  {
    node := &Node{Pre: this.Head, Next: this.Head.Next, Val: val}
    this.cnt++
    next := this.Head.Next
    next.Pre = node
    this.Head.Next = node
}


func (this *MyLinkedList) AddAtTail(val int)  {
    node := &Node{Pre: this.Tail.Pre, Next: this.Tail, Val: val}
    this.cnt++
    pre := this.Tail.Pre
    pre.Next = node
    this.Tail.Pre = node
}


func (this *MyLinkedList) AddAtIndex(index int, val int)  {
    if index == this.cnt {
        this.AddAtTail(val)
        return
    }
    if !this.isValid(index) {
        return 
    }
    pre, cur := this.Head, this.Head.Next
    for i := 0; i < index; i++ {
        pre = cur
        cur = cur.Next
    }
    node := &Node{Pre: pre, Next: cur, Val: val}
    cur.Pre = node
    pre.Next = node
    this.cnt++
}


func (this *MyLinkedList) DeleteAtIndex(index int)  {
    if !this.isValid(index) {
        return
    }
    pre, cur := this.Head, this.Head.Next
    for i := 0; i <= index; i++ {
        pre = cur
        cur = cur.Next
    }
    if pre.Pre != nil {
        pre.Pre.Next = cur
    }
    cur.Pre = pre.Pre
    this.cnt--
}

func (this *MyLinkedList) isValid(index int) bool {
    if index >= this.cnt {
        return false
    } else {
        return true
    }
}


/**
 * Your MyLinkedList object will be instantiated and called as such:
 * obj := Constructor();
 * param_1 := obj.Get(index);
 * obj.AddAtHead(val);
 * obj.AddAtTail(val);
 * obj.AddAtIndex(index,val);
 * obj.DeleteAtIndex(index);
 */
 ```

## [Design Browser History](https://leetcode.com/problems/design-browser-history)

A: 用两个栈分别存储后退和前进历史。

```cpp
class BrowserHistory {
public:
    stack<string> backStk;
    stack<string> forStk;
    BrowserHistory(string homepage) {
        backStk.push(homepage);
    }
    
    void visit(string url) {
        backStk.push(url);
        stack<string>().swap(forStk);
    }
    
    string back(int steps) {
        steps = min(int(backStk.size()), steps);
        for (int i = 0; i < steps && backStk.size() > 1; i++) {
            forStk.push(backStk.top());
            backStk.pop();
        }
        return backStk.top();
    }
    
    string forward(int steps) {
        steps = min(int(forStk.size()), steps);
        for (int i = 0; i < steps; i++) {
            backStk.push(forStk.top());
            forStk.pop();
        }
        return backStk.top();
    }
};

/**
 * Your BrowserHistory object will be instantiated and called as such:
 * BrowserHistory* obj = new BrowserHistory(homepage);
 * obj->visit(url);
 * string param_2 = obj->back(steps);
 * string param_3 = obj->forward(steps);
 */
```

```go
type BrowserHistory struct {
    back []string
    next []string
}


func Constructor(homepage string) BrowserHistory {
    ans := BrowserHistory{ back: []string{homepage}, next: []string{}}
    return ans
}


func (this *BrowserHistory) Visit(url string)  {
    this.next = []string{}
    this.back = append(this.back, url)
}


func (this *BrowserHistory) Back(steps int) string {
    for steps > 0 && len(this.back) > 1 {
        cur := this.back[len(this.back) - 1]
        this.back = this.back[:len(this.back) - 1]
        this.next = append(this.next, cur)
        steps--
    }
    return this.back[len(this.back) - 1]
}


func (this *BrowserHistory) Forward(steps int) string {
    for steps > 0 && len(this.next) > 0 {
        cur := this.next[len(this.next) - 1]
        this.next = this.next[:len(this.next) - 1]
        this.back = append(this.back, cur)
        steps--
    }
    return this.back[len(this.back) - 1]
}


/**
 * Your BrowserHistory object will be instantiated and called as such:
 * obj := Constructor(homepage);
 * obj.Visit(url);
 * param_2 := obj.Back(steps);
 * param_3 := obj.Forward(steps);
 */
```

## [Rotate List](https://leetcode.com/problems/rotate-list)

A: 将链表连成环，新链表的尾指针在len-k处。

```cpp
class Solution {
public:
    ListNode* rotateRight(ListNode* head, int k) {
        if(!head) return head;
        
        int len=1; // number of nodes
        ListNode *newH, *tail;
        newH=tail=head;
        
        while(tail->next)  // get the number of nodes in the list
        {
            tail = tail->next;
            len++;
        }
        tail->next = head; // circle the link

        if(k %= len) 
        {
            for(auto i=0; i<len-k; i++) tail = tail->next; // the tail node is the (len-k)-th node (1st node is head)
        }
        newH = tail->next; 
        tail->next = NULL;
        return newH;
    }
};
```

## [Reverse Linked List II](https://leetcode.com/problems/reverse-linked-list-ii)

A: 找到起始结点和起始节点之前一个节点，反复交换到终止节点。

```cpp
class Solution {
public:
   ListNode* reverseBetween(ListNode* head, int m, int n) {
       ListNode *dummy = new ListNode(0), *pre = dummy, *cur;
       dummy -> next = head;
       for (int i = 0; i < m - 1; i++) {
           pre = pre -> next;
       }
       cur = pre -> next;
       for (int i = 0; i < n - m; i++) {
           ListNode* temp = pre -> next;
           pre -> next = cur -> next;
           cur -> next = cur -> next -> next;
           pre -> next -> next = temp;
       }
       return dummy -> next;
   }
};
```

## [Design Circular Queue](https://leetcode.com/problems/design-circular-queue)

A: 在数组中维护queue。

```cpp
class MyCircularQueue {
public:
    MyCircularQueue(int k) {
        data.resize(k);
        maxSize = k;
    }
    bool enQueue(int val) {
        if (isFull()) return false;
        tail = (tail + 1) % maxSize;
        data[tail] = val;
        return true;
    }
    bool deQueue() {
        if (isEmpty()) return false;
        if (head == tail) head = 0, tail = -1;
        else head = (head + 1) % maxSize;
        return true;
    }
    int Front() {
        return isEmpty() ? -1 : data[head];
    }
    int Rear() {
        return isEmpty() ? -1 : data[tail];
    }
    bool isEmpty() {
        return tail == -1;
    }
    bool isFull() {
        return !isEmpty() && (tail + 1) % maxSize == head;
    }
private:
    int maxSize, head = 0, tail = -1;
    vector<int> data;
};
```

## [Insertion Sort List](https://leetcode.com/problems/insertion-sort-list)

A: 每次从表头开始找cur的插入位置即可。

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* insertionSortList(ListNode* head) {
        ListNode* dummy = new ListNode(0);
        dummy -> next = head;
        ListNode *pre = dummy, *cur = head;
        while (cur) {
            if ((cur -> next) && (cur -> next -> val < cur -> val)) {
                while ((pre -> next) && (pre -> next -> val < cur -> next -> val)) {
                    pre = pre -> next;
                }
                ListNode* temp = pre -> next;
                pre -> next = cur -> next;
                cur -> next = cur -> next -> next;
                pre -> next -> next = temp;
                pre = dummy;
            }
            else {
                cur = cur -> next;
            }
        }
        return dummy -> next;
    }
};
```

## [Remove Linked List Elements](https://leetcode.com/problems/remove-linked-list-elements)

A: 虚拟头结点，按顺序检查。

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* removeElements(ListNode* head, int val) {
        if (!head) return head;
        ListNode *dummy = new ListNode();
        dummy->next = head;
        ListNode *cur = head, *pre = dummy;
        while (cur) {
            if (cur->val == val) {
                pre->next = cur->next;
            } else {
                pre = cur;
            }
            cur = cur->next;
        }
        return dummy->next;
    }
};
```

```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func removeElements(head *ListNode, val int) *ListNode {
    dummy := &ListNode{Val: 0, Next: head}
    if head == nil {
        return head
    }
    var cur, pre *ListNode = head, dummy
    for cur != nil {
        if cur.Val == val {
            pre.Next = cur.Next
        } else {
            pre = cur
        }
        cur = cur.Next
    }
    return dummy.Next
}
```

## [Palindrome Linked List](https://leetcode.com/problems/palindrome-linked-list)

A: 快慢指针找到链表中点，将后半段reverse，然后两边同时遍历并比较。

```cpp
class Solution {
public:
    bool isPalindrome(ListNode* head) {
        ListNode *slow = head, *fast = head, *prev, *temp;
        while (fast && fast->next)
            slow = slow->next, fast = fast->next->next;
        prev = slow, slow = slow->next, prev->next = NULL;
        while (slow)
            temp = slow->next, slow->next = prev, prev = slow, slow = temp;
        fast = head, slow = prev;
        while (slow)
            if (fast->val != slow->val) return false;
            else fast = fast->next, slow = slow->next;
        return true;
    }
};
```

```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func isPalindrome(head *ListNode) bool {
    slow, fast := head, head
    for fast != nil && fast.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
    }
    pre, cur := slow, slow.Next
    pre.Next = nil
    var next *ListNode
    if cur != nil {
        next = cur.Next
    }
    for cur != nil {
        cur.Next = pre
        pre = cur
        cur = next
        if next == nil {
            break
        }
        next = next.Next
    }
    left, right := head, pre
    for left != nil && right != nil && left != right {
        if left.Val != right.Val {
            return false
        } else {
            left = left.Next
            right = right.Next
        }
    }
    return true
}
```

## [Remove Duplicates from Sorted List](https://leetcode.com/problems/remove-duplicates-from-sorted-list)

A: 检测到两个相同节点则delete。

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        ListNode* cur = head;
        while(cur) {
            while(cur->next && cur->val == cur->next->val) {
                cur->next = cur->next->next;
            }
            cur = cur->next;
        }
        return head;
    }
};
```

```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func deleteDuplicates(head *ListNode) *ListNode {
    slow, fast := head, head
    for fast != nil {
        if slow.Val != fast.Val {
            slow.Next = fast
            slow = slow.Next
        }
        fast = fast.Next
    }
    if slow != nil {
        slow.Next = nil
    }
    return head
}
```

## [Intersection of Two Linked Lists](https://leetcode.com/problems/intersection-of-two-linked-lists)

A: 公共长度为C，单独部分长度分别记为A、B。两个链表会同时到达第二次交点（A+C+B或B+C+A）。

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        ListNode *ptrA = headA, *ptrB = headB;
        while (ptrA != ptrB) { 
            ptrA = ptrA ? ptrA->next : headB;
            ptrB = ptrB ? ptrB->next : headA;
        }
        return ptrA;
    }
};
```

```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func getIntersectionNode(headA, headB *ListNode) *ListNode {
    cur1, cur2 := headA, headB
    for cur1 != nil && cur2 != nil && cur1 != cur2 {
        cur1 = cur1.Next
        cur2 = cur2.Next
        if cur1 == cur2 {
            return cur1
        }
        if cur1 == nil {
            cur1 = headB
        }
        if cur2 == nil {
            cur2 = headA
        }
    }
    return cur1
}
```

A: 计算两个链表的长度差，然后长的先走差值步，然后同时走，直到相遇。

```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func getIntersectionNode(headA, headB *ListNode) *ListNode {
    curA, curB := headA, headB
    lenA, lenB := 0, 0
    for curA != nil {
        lenA++
        curA = curA.Next
    }
    for curB != nil {
        lenB++
        curB = curB.Next
    }
    if lenA > lenB {
        lenA, lenB = lenB, lenA
        headA, headB = headB, headA
    }
    gap := lenB - lenA
    for gap > 0 {
        headB = headB.Next
        gap--
    }
    for headA != nil && headB != nil {
        if headA == headB {
            return headA
        }
        headA = headA.Next
        headB = headB.Next
    }
    return nil
}
```

## [Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii)

A: 快慢指针，然后找到相遇点，然后从相遇点和头节点同时走，相遇点即为环的入口。

```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func detectCycle(head *ListNode) *ListNode {
    if head == nil {
        return nil
    }
    slow, fast := head, head
    for fast != nil && fast.Next != nil {
        fast = fast.Next.Next
        slow = slow.Next
        if slow == fast {
            cur1, cur2 := fast, head
            for cur1 != cur2 {
                cur1 = cur1.Next
                cur2 = cur2.Next
            }
            return cur1
        }
    }
    return nil
}
```

## [Middle of the Linked List](https://leetcode.com/problems/middle-of-the-linked-list)

A: 快慢指针。

```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func middleNode(head *ListNode) *ListNode {
    slow, fast := head, head
    for fast.Next != nil && fast.Next.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
    }
    if fast.Next == nil {
        return slow
    } else {
        return slow.Next
    }
}
```

## [Maximum Twin Sum of a Linked List](https://leetcode.com/problems/maximum-twin-sum-of-a-linked-list)

A: 用栈将后半段链表反转，**应该存在其他反转部分链表的方法**。

```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func pairSum(head *ListNode) int {
    ans := 0
    stk := []int{}
    slow, fast := head, head
    for fast != nil && fast.Next != nil {
        fast = fast.Next.Next
        slow = slow.Next
    }
    for slow != nil {
        stk = append(stk, slow.Val)
        slow = slow.Next
    }
    fast = head
    for len(stk) != 0 {
        sum := fast.Val + stk[len(stk) - 1]
        stk = stk[:len(stk) - 1]
        if sum > ans {
            ans = sum
        }
        fast = fast.Next
    }
    return ans
}
```

## [复杂链表的复制](https://www.nowcoder.com/practice/f836b2c43afc4b35ad6adc41ec941dba?tpId=265&tags=&title=&difficulty=0&judgeStatus=0&rp=1&sourceUrl=%2Fexam%2Foj%2Fta%3FtpId%3D13)

A: 将每个节点复制一份，然后**将复制的节点插入到原节点的后面**，然后复制节点的random指针指向原节点的random指针的下一个节点，**最后将复制的节点从原链表中拆分出来**。

```go
func Clone( head *RandomListNode ) *RandomListNode {
    if head == nil {
        return nil
    }
    cur := head
    for cur != nil {
        pClone := CloneNode(cur)
        pre := cur
        cur = cur.Next
        pre.Next = pClone
        pClone.Next = cur
    }
    cur = head
    for cur != nil {
        pClone := cur.Next
        if cur.Random != nil {
            pClone.Random = cur.Random.Next
        }
        cur = cur.Next.Next
    }
    dummy := &RandomListNode{}
    pre, cur := dummy, head
    for cur != nil {
        pre.Next = cur.Next
        pre = pre.Next
        cur.Next = cur.Next.Next
        cur = cur.Next
    }
    return dummy.Next
}

func CloneNode(node *RandomListNode) *RandomListNode {
    return &RandomListNode{
        Label: node.Label,
    }
}
```

## [二叉搜索树与双向链表](https://www.nowcoder.com/practice/947f6eb80d944a84850b0538bf0ec3a5?tpId=265&tags=&title=&difficulty=0&judgeStatus=0&rp=1&sourceUrl=%2Fexam%2Foj%2Fta%3FtpId%3D13)

A: 递归，**递归的返回值是左子树排序的头节点**，然后将左子树的最后一个节点和根节点连接，然后将根节点和右子树的最左节点连接。注意左子树为空时，返回的是根节点。

```go
func Convert( pRootOfTree *TreeNode ) *TreeNode {
    if pRootOfTree == nil {
        return nil
    }
    if pRootOfTree.Left == nil && pRootOfTree.Right == nil {
        return pRootOfTree
    }
    left := Convert(pRootOfTree.Left)
    right := Convert(pRootOfTree.Right)
    var ans *TreeNode
    if left != nil {
        ans = left
    } else {
        ans = pRootOfTree
    }
    for left != nil && left.Right != nil {
        left = left.Right
    }
    if left != nil {
        left.Right = pRootOfTree
        pRootOfTree.Left = left
    }
    if right != nil {
        right.Left = pRootOfTree
        pRootOfTree.Right = right
    }
    return ans
}
```
