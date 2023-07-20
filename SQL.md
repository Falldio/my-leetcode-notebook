# SQL

## [Combine Two Tables](https://leetcode.com/problems/combine-two-tables/)

A: 简单的左连接，注意条件用`ON`，而不是`WHERE`。`WHERE`只能用于过滤，而`ON`可以用于连接条件。

```sql
SELECT
    p.firstName,
    p.lastName,
    a.city,
    a.state
FROM Person p
LEFT JOIN Address a
ON p.personId = a.personId
```

## [Second Highest Salary](https://leetcode.com/problems/second-highest-salary/)

A: 用`IFNULL`函数，如果第二高的工资不存在，就返回`NULL`。`LIMIT`的第一个参数是偏移量，第二个参数是返回的行数。

```sql
SELECT
IFNULL((
    SELECT DISTINCT
        salary
    FROM Employee
    ORDER BY salary DESC
    LIMIT 1, 1
), NULL) AS SecondHighestSalary
```

## [Nth Highest Salary](https://leetcode.com/problems/nth-highest-salary/)

A: `MySQL`函数的写法，`DECLARE`定义变量，`SET`赋值，`RETURN`返回结果。`IFNULL`函数的第二个参数是当第一个参数为`NULL`时的返回值。

```sql
CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
    DECLARE OFF INT;
    SET OFF := N - 1;
    RETURN (
        SELECT (
            IFNULL(
                  (SELECT DISTINCT
                      salary
                  FROM Employee
                  ORDER BY salary DESC
                  LIMIT OFF, 1),
                  NULL
            )
        )
    );
END
```

## [Rank Scores](https://leetcode.com/problems/rank-scores/)

A: 用子查询，先找出所有不同的分数，然后再找出每个分数的排名。注意`COUNT`函数的用法，`COUNT(*)`表示计算行数，`COUNT(S2.Score)`表示计算`S2.Score`列的非空值的个数。`Rank`是一个关键字，所以要用反引号括起来。`FROM`后面有多张表表示的是**笛卡尔积**，所以要用`WHERE`来过滤。

```sql
SELECT S.Score, COUNT(S2.Score) as `Rank`
FROM Scores S,
(SELECT DISTINCT Score FROM Scores) S2
WHERE S.Score<=S2.Score
GROUP BY S.Id
ORDER BY S.Score DESC;
```

A: 用`INNER JOIN`，`INNER JOIN`的条件是`S.score <= T.score`，这样就可以找出所有不同的分数。`COUNT(DISTINCT T.score)`表示计算`T.score`列的不同值的个数。

```sql
SELECT
  S.score,
  COUNT(DISTINCT T.score) AS 'rank'
FROM
  Scores S
  INNER JOIN Scores T ON S.score <= T.score
GROUP BY
  S.id,
  S.score
ORDER BY
  S.score DESC;
```

A: 窗口函数，`DENSE_RANK`表示计算密集排名，`RANK`表示计算排名。`DENSE_RANK`和`RANK`的区别是，`DENSE_RANK`不会跳过相同的分数，而`RANK`会跳过相同的分数。`DENSE_RANK`和`RANK`的区别是，`DENSE_RANK`不会跳过相同的分数，而`RANK`会跳过相同的分数。

```sql
SELECT
  score,
  DENSE_RANK() OVER (ORDER BY score DESC) AS 'rank'
FROM
    Scores
```

## [Consecutive Numbers](https://leetcode.com/problems/consecutive-numbers/)

A: 注意要加入`DISTINCT`，否则会有重复的结果。`DISTNCT`用于过滤重复的行，`DISTINCT`后面可以跟多个列名，表示这些列的组合不能重复。

```sql
SELECT DISTINCT
    l1.num AS ConsecutiveNums
FROM
    Logs as l1,
    Logs as l2,
    Logs as l3
WHERE 
    l1.id = l2.id-1
    AND l2.id = l3.id-1
    AND l1.num = l2.num
    AND l2.num = l3.num
```
