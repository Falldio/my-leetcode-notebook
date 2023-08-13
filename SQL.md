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

A: 窗口函数，`DENSE_RANK`表示计算密集排名，`RANK`表示计算排名。`DENSE_RANK`和`RANK`的区别是，`DENSE_RANK`不会跳过相同的分数，而`RANK`会跳过相同的分数。

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

## [Employees Earning More Than Their Managers](https://leetcode.com/problems/employees-earning-more-than-their-managers/)

A: 自连接，注意`WHERE`和`ON`的区别。`WHERE`用于过滤，`ON`用于连接条件。

```sql
SELECT
    e.name AS Employee
FROM
    Employee AS e
    JOIN Employee AS m
ON e.managerId = m.id
WHERE e.salary > m.salary
```

## [Duplicate Emails](https://leetcode.com/problems/duplicate-emails/)

A: `GROUP BY`和`HAVING`的用法。`GROUP BY`用于分组，`HAVING`用于过滤分组后的结果。

```sql
SELECT
    Email
FROM
    Person
GROUP BY
    Email
HAVING
    COUNT(Email) > 1
```

## [Customers Who Never Order](https://leetcode.com/problems/customers-who-never-order/)

A: `LEFT JOIN`和`IS NULL`的用法。`LEFT JOIN`表示左连接，`IS NULL`表示为空。

```sql
SELECT
    c.name AS Customers
FROM
    Customers AS c
LEFT JOIN Orders AS o
ON c.id = o.customerId
WHERE o.id IS NULL
```

## [Department Highest Salary](https://leetcode.com/problems/department-highest-salary/)

A: 利用max函数，先找出每个部门的最高工资，然后再找出对应的员工。

```sql
SELECT
    d.name AS Department,
    e.name AS Employee,
    e.salary AS Salary
FROM
    Employee AS e,
    Department AS d
WHERE e.departmentId = d.id
    AND (e.departmentId, e.salary) in (
        SELECT
            departmentId,
            max(salary) AS max
        FROM Employee
        GROUP BY departmentId
    )
```

A: 利用窗口函数，先找出每个部门的最高工资，然后再找出对应的员工。`PARTITION BY`表示分组，`ORDER BY`表示排序，`DESC`表示降序，`ASC`表示升序。

```sql
SELECT
    Department,
    Employee,
    Salary
FROM (
    SELECT
        d.name AS Department,
        e.name AS Employee,
        e.salary AS Salary,
        rank() OVER (
            PARTITION BY e.departmentId
            ORDER BY e.salary DESC
        ) AS rn
    FROM Employee AS e
    JOIN Department AS d
    ON e.departmentId = d.Id
) AS tmp
WHERE tmp.rn = 1
```

## [Department Top Three Salaries](https://leetcode.com/problems/department-top-three-salaries/)

A: 使用`DENSE_RANK`函数。

```sql
SELECT
    Department,
    Employee,
    Salary
FROM (
    SELECT
        e.name AS Employee,
        e.salary AS Salary,
        d.name AS Department,
        dense_rank() OVER (
            PARTITION BY d.name
            ORDER BY e.salary DESC
        ) AS rn
    FROM Employee AS e, Department AS d
    WHERE e.departmentId = d.id
) AS t
WHERE rn <= 3
```

## [Delete Duplicate Emails](https://leetcode.com/problems/delete-duplicate-emails/)

A: `DELETE`可以和`JOIN`一起使用。

```sql
DELETE p1
FROM Person p1, Person p2
WHERE p1.Email = p2.Email AND p1.Id > p2.Id
```

## [Rising Temperature](https://leetcode.com/problems/rising-temperature)

A: 注意MySQL中的日期函数。`DATEDIFF`表示两个日期之间的天数差，`DATE_SUB`表示日期减去一定的天数，`DATE_ADD`表示日期加上一定的天数。

```sql
SELECT
    w1.id AS id
FROM
    Weather AS w1,
    Weather AS w2
WHERE DATEDIFF(w1.recordDate, w2.recordDate) = 1
    AND w1.temperature > w2.temperature
```

## [Trips and Users](https://leetcode.com/problems/trips-and-users/)

A: 注意`ROUND`函数的用法，`ROUND`函数用于四舍五入，第二个参数表示保留几位小数。`COUNT`不统计`NULL`值。

```sql
SELECT t.request_at AS Day,
       ROUND(
           (COUNT(
               IF(
                   t.status != 'completed',
                   TRUE,
                   NULL
                )
            ) / COUNT(*)), 2
        ) AS 'Cancellation Rate'
FROM Trips AS t
WHERE t.client_Id IN (Select users_Id from Users where Banned='No') 
    AND t.driver_Id IN (Select users_Id from Users where Banned='No')
    AND t.request_at BETWEEN '2013-10-01' AND '2013-10-03'
GROUP by t.request_at
```

## [Game Play Analysis I](https://leetcode.com/problems/game-play-analysis-i/)

A: 窗口函数

```sql
SELECT DISTINCT
    player_id,
    MIN(event_date) OVER(PARTITION BY player_id) AS first_login
FROM
    Activity
```

## [Game Play Analysis IV](https://leetcode.com/problems/game-play-analysis-iv/)

A: 利用`DATE_SUB`函数在Activity表中查询是否存在连续第二天登陆的情况，统计这种玩家的个数为分子，再统计所有玩家的个数为分母。

```sql
SELECT
  ROUND(COUNT(DISTINCT player_id) / (SELECT COUNT(DISTINCT player_id) FROM Activity), 2) AS fraction
FROM
  Activity
WHERE
  (player_id, DATE_SUB(event_date, INTERVAL 1 DAY))
  IN (
    SELECT player_id, MIN(event_date) AS first_login FROM Activity GROUP BY player_id
  )
```

## [Managers with at Least 5 Direct Reports](https://leetcode.com/problems/managers-with-at-least-5-direct-reports/)

A: `INNER JOIN`可以简写为`JOIN`，`ON`表示连接条件，`GROUP BY`表示分组，`HAVING`表示过滤分组后的结果。

```sql
SELECT m.name
FROM Employee as e
INNER JOIN Employee as m
On e.managerId=m.id
GROUP BY e.managerId 
HAVING count(e.id)>=5
```

## [Employee Bonus](https://leetcode.com/problems/employee-bonus/)

A: 使用左连接，确保bonus为NULL的员工也能被查询出来。

```sql
SELECT
    name,
    bonus
FROM
    Employee e
LEFT JOIN Bonus b
ON e.empId = b.empId
WHERE bonus < 1000
    OR bonus IS NULL
```

## [Investments in 2016](https://leetcode.com/problems/investments-in-2016/)

A: 可以通过统计2015和地理位置的次数来进行过滤。

```sql
select
    round(sum(tiv_2016), 2) as tiv_2016
from
    (
        select
            tiv_2016
            , count(*) over (partition by tiv_2015) as tiv_2015_cnt
            , count(*) over (partition by lat, lon) as location_cnt
        from
            insurance
    ) t 
where tiv_2015_cnt > 1 and location_cnt = 1
```

## [Classes More Than 5 Students](https://leetcode.com/problems/classes-more-than-5-students/description/)

A: 在HAVING中可以添加限定条件。

```sql
SELECT
    class
FROM
    Courses
GROUP BY class
HAVING COUNT(student) >= 5
```
