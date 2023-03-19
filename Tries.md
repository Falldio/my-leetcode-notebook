# Tries

## [Implement Trie Prefix Tree](https://leetcode.com/problems/implement-trie-prefix-tree)

```cpp
class TrieNode {
public:
    TrieNode* children[26];
    bool isWord;
    
    TrieNode() {
        for (int i = 0; i < 26; i++) {
            children[i] = nullptr;
        }
        isWord = false;
    }
};

class Trie {
public:
    Trie() {
        root = new TrieNode();
    }
    
    void insert(string word) {
        TrieNode* node = root;
        int curr = 0;
        
        for (int i = 0; i < word.size(); i++) {
            curr = word[i] - 'a';
            if (node->children[curr] == nullptr) {
                node->children[curr] = new TrieNode();
            }
            node = node->children[curr];
        }
        
        node->isWord = true;
    }
    
    bool search(string word) {
        TrieNode* node = root;
        int curr = 0;
        
        for (int i = 0; i < word.size(); i++) {
            curr = word[i] - 'a';
            if (node->children[curr] == nullptr) {
                return false;
            }
            node = node->children[curr];
        }
        
        return node->isWord;
    }
    
    bool startsWith(string prefix) {
        TrieNode* node = root;
        int curr = 0;
        
        for (int i = 0; i < prefix.size(); i++) {
            curr = prefix[i] - 'a';
            if (node->children[curr] == nullptr) {
                return false;
            }
            node = node->children[curr];
        }
        
        return true;
    }
private:
    TrieNode* root;
};

/**
 * Your Trie object will be instantiated and called as such:
 * Trie* obj = new Trie();
 * obj->insert(word);
 * bool param_2 = obj->search(word);
 * bool param_3 = obj->startsWith(prefix);
 */
```

```go
type Node struct {
    isWord bool
    nodes []*Node
}

type Trie struct {
    root *Node
}


func Constructor() Trie {
    return Trie{root: &Node{}}
}


func (this *Trie) Insert(word string)  {
    cur := this.root
    for _, ch := range word {
        if cur.nodes == nil {
            cur.nodes = make([]*Node, 26)
        }
        if cur.nodes[ch - 'a'] == nil {
            next := &Node{}
            cur.nodes[ch - 'a'] = next
        }
        cur = cur.nodes[ch - 'a']
    }
    cur.isWord = true
}


func (this *Trie) Search(word string) bool {
    cur := this.root
    for _, ch := range word {
        if cur.nodes == nil || cur.nodes[ch - 'a'] == nil {
            return false
        }
        cur = cur.nodes[ch - 'a']
    }
    return cur.isWord
}


func (this *Trie) StartsWith(prefix string) bool {
    cur := this.root
    for _, ch := range prefix {
        if cur.nodes == nil || cur.nodes[ch - 'a'] == nil {
            return false
        }
        cur = cur.nodes[ch - 'a']
    }
    return true
}


/**
 * Your Trie object will be instantiated and called as such:
 * obj := Constructor();
 * obj.Insert(word);
 * param_2 := obj.Search(word);
 * param_3 := obj.StartsWith(prefix);
 */
 ```

## [Design Add And Search Words Data Structure](https://leetcode.com/problems/design-add-and-search-words-data-structure)

A: 用int标识当前处理位置。

```cpp
class TrieNode {
public:
    TrieNode* children[26];
    bool isWord;
    
    TrieNode() {
        for (int i = 0; i < 26; i++) {
            children[i] = NULL;
        }
        isWord = false;
    }
};

class WordDictionary {
public:
    WordDictionary() {
        root = new TrieNode();
    }
    
    void addWord(string word) {
        TrieNode* node = root;
        int curr = 0;
        
        for (int i = 0; i < word.size(); i++) {
            curr = word[i] - 'a';
            if (node->children[curr] == nullptr) {
                node->children[curr] = new TrieNode();
            }
            node = node->children[curr];
        }
        
        node->isWord = true;
    }
    
    bool search(string word) {
        TrieNode* node = root;
        return searchInNode(word, 0, node);
    }
private:
    TrieNode* root;
    
    bool searchInNode(string& word, int i, TrieNode* node) {
        if (!node) {
            return false;
        }
        if (i == word.size()) {
            return node->isWord;
        }
        if (word[i] != '.') {
            return searchInNode(word, i + 1, node->children[word[i] - 'a']);
        }
        for (int j = 0; j < 26; j++) {
            if (searchInNode(word, i + 1, node->children[j])) {
                return true;
            }
        }
        return false;
    }
};


/**
 * Your WordDictionary object will be instantiated and called as such:
 * WordDictionary* obj = new WordDictionary();
 * obj->addWord(word);
 * bool param_2 = obj->search(word);
 */
 ```

 ```go
 type Node struct {
    isWord bool
    next []*Node
}

type WordDictionary struct {
    isWord bool
    next []*WordDictionary
}


func Constructor() WordDictionary {
    return WordDictionary{next: make([]*WordDictionary, 26)}
}


func (this *WordDictionary) AddWord(word string)  {
    cur := this
    for i := 0; i < len(word); i++ {
        if cur.next == nil {
            cur.next = make([]*WordDictionary, 26)
        }
        if cur.next[word[i] - 'a'] == nil {
            cur.next[word[i] - 'a'] = &WordDictionary{next: make([]*WordDictionary, 26)}
        }
        cur = cur.next[word[i] - 'a']
    }
    cur.isWord = true
}


func (this *WordDictionary) Search(word string) bool {
    cur := this
    if len(word) == 0 {
        return cur.isWord
    }
    if word[0] == '.' {
        for _, node := range cur.next {
            if node != nil {
                if node.Search(word[1:]) {
                    return true
                }
            }
        }
    } else {
        if cur.next[word[0] - 'a'] == nil {
            return false
        } else {
            return cur.next[word[0] - 'a'].Search(word[1:])
        }
    }
    return false
}


/**
 * Your WordDictionary object will be instantiated and called as such:
 * obj := Constructor();
 * obj.AddWord(word);
 * param_2 := obj.Search(word);
 */
```

## [Word Search II](https://leetcode.com/problems/word-search-ii)

A: Trie，DFS，回溯。

 ```cpp
class TrieNode {
public:
    TrieNode* children[26];
    bool isWord;
    
    TrieNode() {
        for (int i = 0; i < 26; i++) {
            children[i] = NULL;
        }
        isWord = false;
    }
};

class Solution {
public:
    vector<string> findWords(vector<vector<char>>& board, vector<string>& words) {
        for (int i = 0; i < words.size(); i++) {
            insert(words[i]);
        }
        
        int m = board.size();
        int n = board[0].size();
        
        TrieNode* node = root;
        vector<string> result;
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                search(board, i, j, m, n, node, "", result);
            }
        }
        
        return result;
    }
private:
    TrieNode* root = new TrieNode();
    
    void insert(string word) {
        TrieNode* node = root;
        int curr = 0;
        
        for (int i = 0; i < word.size(); i++) {
            curr = word[i] - 'a';
            if (node->children[curr] == NULL) {
                node->children[curr] = new TrieNode();
            }
            node = node->children[curr];
        }
        
        node->isWord = true;
    }
    
    void search(vector<vector<char>>& board, int i, int j, int m, int n, TrieNode* node, string word, vector<string>& result) {
        if (i < 0 || i >= m || j < 0 || j >= n || board[i][j] == '#') {
            return;
        }
        
        char c = board[i][j];
        
        node = node->children[c - 'a'];
        if (node == NULL) {
            return;
        }
        
        word += board[i][j];
        if (node->isWord) {
            result.push_back(word);
            node->isWord = false;
        }
        
        board[i][j] = '#';
        
        search(board, i - 1, j, m, n, node, word, result);
        search(board, i + 1, j, m, n, node, word, result);
        search(board, i, j - 1, m, n, node, word, result);
        search(board, i, j + 1, m, n, node, word, result);
        
        board[i][j] = c;
    }
};
```