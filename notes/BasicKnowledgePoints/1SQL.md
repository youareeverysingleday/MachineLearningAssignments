# 基本SQL相关

## 涉及的章节

completed.

## 笔记

### 1. SQL基础

1. SQL的3种语句类：
   1. DML Data Manipulation Language 数据操纵语言，用于查询和修改记录。
   2. DDL Data Definition Language 数据定义语言，用于定义数据库的结构，比如创建、修改或者删除数据库对象。
   3. DCL Data Control Language 数据控制语言，用于控制数据库的访问。
2. 常用命令
   1. mysql -uusername -ppassword;在terminal中进入mysql命令行。
   2. use dbname;指定使用数据库。
   3. show tables;在当前数据库中显示所有表。
   4. ctrl+l清除屏幕
   5. exit退出mysql命令行界面，注意没有分号结尾。
   6. start-all.sh在terminal中启动Hive。
   7. hive -S启动Hive。
3. Hive是Hadoop体系中的一个数据分析引擎。Hive也支持SQL语句。
4. [在线SQL语句测试](http://sqlfiddle.com/)。
5. select语句
   1. select语句中可以使用distinct关键字来去重。[参考](https://www.runoob.com/sql/sql-distinct.html)
   2. 通过desc tablename;来查看表结构。
   3. select中支持表达式。
6. 子查询分为单行子查询和多行子查询。
   1. 单行子查询只能使用单行操作符。
      1. 只返回一行。
      2. 可以使用单行比较操作符，比如：=, >, >=, <, <=, >.
   2. 多行子查询智能使用多行操作符。
   3. 主查询和子查询是1对多的关系。举例如下：

   ```SQL
   select ename, job, sal
   from emp
   where job = (select job 
               from emp
               where empno = 7566)
   and sal > (select sal
            from emp
            where empno=7782);
   ```

### 2. SQL高级查询

Hive是Hadoop的SQL引擎。也就是在Hive中可以使用SQL来查询大数据的内容。相对于Hadoop对于SQL的接口。

1. 笛卡尔积与多表查询
2. SQL的函数
   1. 单行函数和多行函数。
   2. 字符函数与数值函数。
   3. 日期函数的使用。
   4. 条件函数的使用。
   5. 正则表达式。
3. SQL函数的基本分类
   1. 单行函数
      1. 字符函数，操作字符串
      2. 数值函数，操作数值。
      3. 日期函数，操作日期。
      4. 条件函数，实现if-else功能。
      5. 正则表达式
      6. 其他函数
   2. 聚合函数
