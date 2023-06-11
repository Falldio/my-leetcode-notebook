# System Design

## [LRU Cache](https://leetcode.com/problems/lru-cache)

A: 链表存储使用记录，将最近操作元素转移到链表前列。

[`splice()`](https://www.cainiaojc.com/cpp/cpp-list-splice-function.html): 将列表插入源列表指定位置中。

```cpp
class LRUCache{
    size_t m_capacity;
    unordered_map<int,  list<pair<int, int>>::iterator> m_map; //m_map_iter->first: key, m_map_iter->second: list iterator;
    list<pair<int, int>> m_list;                               //m_list_iter->first: key, m_list_iter->second: value;
public:
    LRUCache(size_t capacity):m_capacity(capacity) {
    }
    int get(int key) {
        auto found_iter = m_map.find(key);
        if (found_iter == m_map.end()) //key doesn't exist
            return -1;
        m_list.splice(m_list.begin(), m_list, found_iter->second); //move the node corresponding to key to front
        return found_iter->second->second;                         //return value of the node
    }
    void put(int key, int value) {
        auto found_iter = m_map.find(key);
        if (found_iter != m_map.end()) //key exists
        {
            m_list.splice(m_list.begin(), m_list, found_iter->second); //move the node corresponding to key to front
            found_iter->second->second = value;                        //update value of the node
            return;
        }
        if (m_map.size() == m_capacity) //reached capacity
        {
           int key_to_del = m_list.back().first; 
           m_list.pop_back();            //remove node in list;
           m_map.erase(key_to_del);      //remove key in map
        }
        m_list.emplace_front(key, value);  //create new node in list
        m_map[key] = m_list.begin();       //create correspondence between key and node
    }
};
```

```go
import "container/list"


type kv struct {
    // key is only needed to delete entry in data map
    // when removing LRU item
    k int
    v int
}

type LRUCache struct {
    capacity int
    data map[int]*list.Element
    hits *list.List
}

func Constructor(capacity int) LRUCache {
    return LRUCache{
        capacity: capacity,
        data: make(map[int]*list.Element),
        hits: list.New(),
    }
}

func (c *LRUCache) Get(key int) int {
    if el, ok := c.data[key]; ok {
        c.hits.MoveToFront(el)
        return el.Value.(kv).v
    } 
    return -1
}


func (c *LRUCache) Put(key int, value int)  {
    // just an update?
    if el, ok := c.data[key]; ok {
        el.Value = kv{k: key, v: value}
        c.hits.MoveToFront(el)
        return
    }
    // full?
    if c.hits.Len() == c.capacity {
        last := c.hits.Back()
        delete(c.data, last.Value.(kv).k)
        c.hits.Remove(last)
    }
    // put
    c.data[key] = c.hits.PushFront(kv{k: key, v: value})
}
```

## [LFU Cache](https://leetcode.com/problems/lfu-cache)

A: 用三个哈希表分别存储{key, value, freq}, {key, 同频率key list}, {freq, key list}。当溢出时，通过min freq找到对应key list，删除其中的一个元素。

```cpp
class LFUCache {
    int cap;
    int size;
    int minFreq;
    unordered_map<int, pair<int, int>> m; //key to {value,freq};
    unordered_map<int, list<int>::iterator> mIter; //key to list iterator;
    unordered_map<int, list<int>>  fm;  //freq to key list; 相同频率下可能有多个key
public:
    LFUCache(int capacity) {
        cap=capacity;
        size=0;
    }
    
    int get(int key) {
        if(m.count(key)==0) return -1;
        
        fm[m[key].second].erase(mIter[key]);
        m[key].second++;
        fm[m[key].second].push_back(key);
        mIter[key]=--fm[m[key].second].end();
        
        if(fm[minFreq].size()==0 ) 
              minFreq++;
        
        return m[key].first;
    }
    
   void put(int key, int value) {
        if(cap<=0) return;
        
        int storedValue=get(key);
        if(storedValue!=-1)
        {
            m[key].first=value;
            return;
        }
        
        if(size>=cap)
        {
            m.erase( fm[minFreq].front() );
            mIter.erase( fm[minFreq].front() );
            fm[minFreq].pop_front();
            size--;
        }
        
        m[key]={value, 1};
        fm[1].push_back(key);
        mIter[key]=--fm[1].end();
        minFreq=1;
        size++;
    }
};
```

```go
import "container/list"


type LFUCache struct {
    cap int
    minF int
    mKV map[int]int
    mKF map[int]int
    mFK map[int]*list.List
}


type Node struct {
    k, v, f int
}


func Constructor(capacity int) LFUCache {
    return LFUCache {
        cap: capacity,
        minF: 0,
        mKV: make(map[int]int),
        mKF: make(map[int]int),
        mFK: make(map[int]*list.List),
    }
}


func (this *LFUCache) Get(key int) int {
    if v, ok := this.mKV[key]; ok {
        this.increaseFreq(key)
        return v
    }
    return -1
}


func (this *LFUCache) Put(key int, value int)  {
    if _, ok := this.mKV[key]; ok {
        this.mKV[key] = value
        this.increaseFreq(key)
        return
    }
    if len(this.mKV) == this.cap {
        fmt.Println(this.minF, key, value)
        fmt.Println(this.mFK[this.minF])
        this.removeMinFreq()
    }
    this.mKV[key] = value
    this.mKF[key] = 1
    if _, ok := this.mFK[1]; !ok {
        this.mFK[1] = list.New()
    }
    this.mFK[1].PushFront(key)
    this.minF = 1
}


func (this *LFUCache) increaseFreq(key int) {
    f := this.mKF[key]
    this.mKF[key]++
    for e := this.mFK[f].Front(); e != nil; e = e.Next() {
        if e.Value.(int) == key {
            this.mFK[f].Remove(e)
            break
        }
    }
    if this.mFK[f].Len() == 0 {
        delete(this.mFK, f)
        if f == this.minF {
            this.minF++
        }
    }
    if l, ok := this.mFK[f + 1]; ok {
        l.PushFront(key)
    } else {
        this.mFK[f + 1] = list.New()
        this.mFK[f + 1].PushFront(key)
    }
}


func (this *LFUCache) removeMinFreq() {
    l := this.mFK[this.minF]
    k := l.Remove(l.Back()).(int)
    if l.Len() == 0 {
        delete(this.mFK, this.minF)
    }
    delete(this.mKV, k)
    delete(this.mKF, k)
}
/**
 * Your LFUCache object will be instantiated and called as such:
 * obj := Constructor(capacity);
 * param_1 := obj.Get(key);
 * obj.Put(key,value);
 */
```

## [Design HashSet](https://leetcode.com/problems/design-hashset)

```go
const ListCapacity = 10
const HashSize = 4

type MyHashSet struct {
	hashSetData
}

func Constructor() MyHashSet {
	return MyHashSet{}
}

func (m *MyHashSet) Add(key int) {
	_ = m.add(key, 0)
}

func (m *MyHashSet) Remove(key int) {
	_ = m.remove(key, 0)
}

func (m *MyHashSet) Contains(key int) bool {
	return m.contains(key, 0)
}

type hashSetData struct {
	size int
	list []int
	next []MyHashSet
}

func hash(key int, count uint) int {
	return key >> (HashSize * count) & (1<<HashSize - 1)
}

func initList() []int {
	return make([]int, 0, ListCapacity)
}

func initNext() []MyHashSet {
	return make([]MyHashSet, 1<<HashSize)
}

func (h *hashSetData) add(key int, count uint) int {
	var added int
	if h.next != nil {
		added = h.next[hash(key, count)].add(key, count+1)
	} else if len(h.list) != ListCapacity {
		if h.list == nil {
			h.list = initList()
		}
		for _, v := range h.list {
			if v == key {
				return 0
			}
		}
		h.list = append(h.list, key)
		added = 1
	} else {
		h.next = initNext()
		for _, v := range h.list {
			_ = h.next[hash(v, count)].add(v, count+1)
		}
		added = h.next[hash(key, count)].add(key, count+1)
		h.list = nil
	}
	h.size += added
	return added
}

func (h *hashSetData) remove(key int, count uint) int {
	var removed int
	if h.next != nil {
		removed = h.next[hash(key, count)].remove(key, count+1)
	} else {
		for i, v := range h.list {
			if v == key {
				removed = 1
			} else if removed != 0 {
				h.list[i-1] = v
			}
		}
		h.list = h.list[:h.size-removed]
	}
	h.size -= removed
	if h.size == 0 {
		h.list = nil
		h.next = nil
	}
	return removed
}

func (h *hashSetData) contains(key int, count uint) bool {
	if h.next != nil {
		return h.next[hash(key, count)].contains(key, count+1)
	}
	for _, v := range h.list {
		if v == key {
			return true
		}
	}
	return false
}
```

## [Design Underground System](https://leetcode.com/problems/design-underground-system)

A: 用interface{}存储**任意类型**的值

```go
type UndergroundSystem struct {
    checkInDB map[int][2]interface{} // id -> [stationName, t]
    checkOutDB map[[2]string][2]int // [checkInStation, checkOutStation] -> [totalTime, count]
}

func Constructor() UndergroundSystem {
    var system UndergroundSystem
    system.checkInDB = map[int][2]interface{}{}
    system.checkOutDB = map[[2]string][2]int{}
    return system
}

func (this *UndergroundSystem) CheckIn(id int, stationName string, t int)  {
    this.checkInDB[id] = [2]interface{}{stationName, t}
}

func (this *UndergroundSystem) CheckOut(id int, stationName string, t int)  {
    checkInDetails := this.checkInDB[id]
    route := [2]string{}
    
    if checkInStation, ok := checkInDetails[0].(string); ok {
        route[0] = checkInStation
        route[1] = stationName
    }
    if checkInTime, ok := checkInDetails[1].(int); ok {
        values := this.checkOutDB[route]
        values[0] += (t - checkInTime)
        values[1] += 1
        this.checkOutDB[route] = values
    }
}

func (this *UndergroundSystem) GetAverageTime(startStation string, endStation string) float64 {
    route := [2]string{startStation, endStation}
    details := this.checkOutDB[route]
    time, freq := float64(details[0]), float64(details[1])
    return time / freq
}
```

## [Snapshot Array](https://leetcode.com/problems/snapshot-array)

A: 记录每个元素的历史值，每次set时，如果当前版本小于快照版本，则添加新的快照，否则更新当前快照的值，最后使用二分查找找到最近的快照版本。

```go
type SnapshotArray struct {
    nums [][][]int // nums, snap version, {snap id, val}
    cnt int // snap cnt
}


func Constructor(length int) SnapshotArray {
    sa := SnapshotArray{
        nums: make([][][]int, length),
        cnt: 0,
    }
    for i := range sa.nums {
        sa.nums[i] = make([][]int, 1)
        sa.nums[i][0] = []int{0, 0}
    }
    return sa
}


func (this *SnapshotArray) Set(index int, val int)  {
    end := len(this.nums[index]) - 1
    snapVer := this.nums[index][end][0]
    if snapVer < this.cnt {
        this.nums[index] = append(this.nums[index], []int{this.cnt, val})
    } else {
        this.nums[index][end][0] = this.cnt
        this.nums[index][end][1] = val
    }
}


func (this *SnapshotArray) Snap() int {
    this.cnt++
    return this.cnt - 1
}


func (this *SnapshotArray) Get(index int, snap_id int) int {
    vers := this.nums[index]
    l, r := 0, len(vers) - 1
    ans := -1
    for l <= r {
        mid := l + (r - l) / 2
        id := vers[mid][0]
        if id == snap_id {
            return vers[mid][1]
        } else if id > snap_id {
            r = mid - 1
        } else {
            l = mid + 1
            ans = vers[mid][1]
        }
    }
    return ans
}


/**
 * Your SnapshotArray object will be instantiated and called as such:
 * obj := Constructor(length);
 * obj.Set(index,val);
 * param_2 := obj.Snap();
 * param_3 := obj.Get(index,snap_id);
 */
 ```
 