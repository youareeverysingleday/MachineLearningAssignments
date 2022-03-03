# python 基础知识

## 1 Python编程基础概念

1. 5种基本对象类型。
   1. 字符串：string，简记为str
   2. 整数：integer，简记为int
      1. 十进制
      2. 八进制：0o100, -0o100
      3. 十六进制：0x40, -0x40
      4. 二进制：0b100000, -0b100000
   3. 浮点数：float
   4. 布尔型：boolean，简记为bool，True, False。
   5. 复数：complex，1+2j, -1-1j
2. type()对象(函数)，用于查看对象的类型。不同类型对象的运算规则不同。不同类型的在计算机中的存储形式不一样。
3. 基础运算符
   1. 算术运算符：+,-,*,/,//,%,\*\*。**其中"//"表示的除法是整数除法，也就是小学时候使用的除法，结果得到商**，不包含余数。其中"\*\*"表示的是指数运算。
   2. 关系运算符：==, !=, >=, >, <, <=。
   3. 逻辑运算符：and, or, not。
4. 运算符优先级
5. input和print对象。input用于接收从命令行（键盘）中传来的数据。

   ```python
   radius = float(input("please input radius of circule:"))
   print(1, 2, 3, seq=',', end='\n') # print(value,..., seq=',', end='\n')，seq由于表示输出多个值时之间的分隔符。end表示输出最后的结尾用什么。
   ```

6. 内置函数和内置数学库
   1. range函数。range函数可以接受3个参数，（下限值，上限值，步长），函数会范围一个从下限值开始，到(上限值-1)结束的range对象，该对象中迭代至直接的差值等于步长。如果省略步长，则默认取1；下限值如果省略，那么默认值取0。

   ```python
   # 由于range返回的是range对象，所以需要使用list将其转换为可以输出的类型。completed 1.7 
   list(range(10))
   list(range(1, 10))
   list(range(1, 10, 2))
   ```

7. 程序语句结构
   1. 顺序结构
   2. 选择结构
   3. 循环结构
8. if-else三元表达式

   ```python
   # 下面两个代码逻辑上等价
   # 1. 这是三元表达式。
   maxn = (n if n>m else m)

   # 2. 这是if-else语句。
   if n>m:
      maxn = n
   else:
      maxn = m
   ```

