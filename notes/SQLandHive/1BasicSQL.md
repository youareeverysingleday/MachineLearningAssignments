# 基本SQL相关

## 涉及的章节

chapter 0.1-0.4

## 笔记

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