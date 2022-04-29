# 线性代数

## 线性代数公式

[课程所有公式总结参考](https://www.renrendoc.com/paper/99978252.html)这个总结并不好。

在机器学习中10个常用公式：

1. $\vec{a}, \vec{x} \in \mathbb{R^{n\times 1}}, then \frac{d\vec{a}^T\vec{x}}{d\vec{x}}=\vec{a}, \frac{d\vec{x}^T\vec{a}}{d\vec{x}}=\vec{a}$
2. $\vec{x} \in \mathbb{R^{n\times 1}}, then \frac{d\vec{x}^T \vec{x}}{d\vec{x}}=2\vec{x}$
3. $\vec{x} \in \mathbb{R^{n\times 1}}, \boldsymbol{y}(\vec{x}) \in \mathbb{R^{m\times 1}}, then \frac{d\boldsymbol{y}^{T}(\vec{x})}{d\vec{x}} = (\frac{d\boldsymbol{y}(\vec{x})}{d\vec{x}^T})^T$
4. $\boldsymbol{A} \in \mathbb{R^{m\times n}}, \vec{x} \in\mathbb{R^{n\times 1}}, then \frac{d\boldsymbol{A}\vec{x}}{d\vec{x}^T}=\boldsymbol{A}$
5. $\boldsymbol{A} \in \mathbb{R^{m\times n}}, \vec{x} \in\mathbb{R^{n\times 1}}, then \frac{d\vec{x}^T\boldsymbol{A}^T}{d\vec{x}}=\boldsymbol{A}^T$
6. $\boldsymbol{A} \in \mathbb{R^{n\times n}} \text{，A是方阵}, \vec{x} \in\mathbb{R^{n\times 1}}, then \frac{d\vec{x}^T\boldsymbol{A}\vec{x}}{d\vec{x}}=(\boldsymbol{A} + \boldsymbol{A}^T)\vec{x}$
7. $\boldsymbol{x} \in \mathbb{R^{m\times n}}, \vec{a} \in \mathbb{R^{m\times 1}}, \vec{b} \in \mathbb{R^{n\times 1}}, then \frac{d\vec{a}^T\boldsymbol{x}\vec{b}}{d\vec{x}}=\vec{a} \vec{b}^T$
8. $\boldsymbol{x} \in \mathbb{R^{n\times m}} \text{注意这里是n}\times \text{m维度。而不是前面的m}\times \text{n维度}, \vec{a} \in \mathbb{R^{m\times 1}}, \vec{b} \in \mathbb{R^{n\times 1}}, then \frac{d\vec{a}^T\boldsymbol{x}^T\vec{b}}{d\vec{x}}=\vec{b}\vec{a}^T$
9. $\boldsymbol{X} \in \mathbb{R^{m\times n}}, \boldsymbol{B} \in \mathbb{R^{n\times m}}, then \frac{d(tr\boldsymbol{X}\boldsymbol{B})}{d\boldsymbol{X}}=\boldsymbol{B}^T$
10. $\boldsymbol{X} \in \mathbb{R^{m\times n}}, \boldsymbol{X} \text{是可逆的。} then \frac{d|\boldsymbol{X}|}{d\boldsymbol{X}}=|\boldsymbol{X}|(\boldsymbol{X}^{-1})^T$

## 1 行列式

### 1.1 需要注意的关键点

1. 行列式是方阵。
2. 在二维空间中，行列式是计算的的是面积；在三维空间中，行列式计算的就是体积。

### 1.2 主要方法

1. 行列式的计算是通过行与行（或者列与列）之间的加减法来完成化简的。主要利用的是10. 11. 12. 13.对应的定义和性质。
2. 行列式等于其主对角线上分块行列式之积。

### 1.3 定义、性质

1. 把n个不同的元素排列为一列，叫做这n个元素的全排列（也简称排列）。
2. 定义标准次序之后，当某一个元素的先后次序与标准次序不同的时候它就构成一个逆序。
3. 一个排列中所有逆序的总数叫做这个排列的逆序数。定义逆序数就是为了计算行列式中的项前面的符号做准备的。
4. 逆序数为奇数的排列叫做奇排列，逆序数为偶数的排列叫做偶排列。
5. 定理1 一个排列中的任意两个元素对换，排列改变奇偶性。
6. 推论 奇排列对换成标准排列的对换次数为奇数，偶排列对换成标准排列的对换次数为偶数。
7. 定义2 设有$n^2$个数，排列成$n$行$n$列的数表  
    $$
    \begin{matrix}
    a_{11} & a_{12} & \cdots & a_{1n} \\
    a_{21} & a_{22} & \cdots & a_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{n1} & a_{n2} & \cdots & a_{nn}
    \end{matrix}
    $$
    做出表中位于不同行不同列的n个数的乘积，并冠以符号$(-1)^t$，得到形如  
    $$(-1)^t a_{1p_1}a_{2p_2}\cdots a_{np_n} \tag{7}$$
    其实就是看下标的逆序数来决定前面的该项前面的符号。一共有两个下标。先固定i的下标，然后确定j的逆序数，这个逆序数就是t。
    的项，其中$p_1p_2\cdots p_n$为自然数$1,2,\cdots ,n$的一个排列，t为这个排列的逆序数。由于这项的排列共有$n!$个，因而形如$(7)$ 式的项共有$n!$项，所有这$n!$项的代数和为  
    $$\sum (-1)^ta_{1p_1}a_{2p_2}\cdots a_{np_n}$$
    成为n阶行列式，记作  
    $$
    D=\begin{vmatrix}
    a_{11} & a_{12} & \cdots & a_{1n} \\
    a_{21} & a_{22} & \cdots & a_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{n1} & a_{n2} & \cdots & a_{nn}
    \end{vmatrix}
    $$  
    简记作$det(a_{ij})$，其中数$a_{ij}$为行列式D的$(i,j)$元。

8. 主对角线以下或者以上的元素都为0的杭历史叫做上或者下三角形行列式，特别是处理对角线元素之外全为0的杭历史叫做对角行列式。
9. 性质1 $D=D^T$。行列式与他的转置行列式相等。
10. 性质2 对换行列式的两行或者两列，行列式变号。
11. 推论 如果行列式两行或者两列完全相同，则此行列式等于0。
12. 性质3 行李额是的某一行或者列中所有的元素都乘以同一个数k，等于用数k乘以此行列式。
13. 推论 行列式中某一行或者列的所有元素的公因子可以提到行列式记号的外面。
    $$
    D=\begin{vmatrix}
    a_{11} & a_{12} & \cdots & a_{1n} \\
    \vdots & \vdots & \ddots & \vdots \\
    ka_{i1} & ka_{i2} & \cdots & ka_{in} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{n1} & a_{n2} & \cdots & a_{nn}
    \end{vmatrix}=kD=k\begin{vmatrix}
    a_{11} & a_{12} & \cdots & a_{1n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{i1} & a_{i2} & \cdots & a_{in} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{n1} & a_{n2} & \cdots & a_{nn}
    \end{vmatrix}
    $$
14. 行列式中如果有两行或者列元素成比例，则此行列式等于0。
15. 做行列式的某一行或者列的元素都是两数之和，例如第i行的元素都是两数之和：  
    $$
    D=\begin{vmatrix}
    a_{11} & a_{12} & \cdots & a_{1n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{i1}+a^{'}_{i1} & a_{i2}+a^{'}_{i2} & \cdots & a_{in}+a^{'}_{in} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{n1} & a_{n2} & \cdots & a_{nn}
    \end{vmatrix}\\
    \text{则D等于下列两个行列式之和：}\\
    D=\begin{vmatrix}
    a_{11} & a_{12} & \cdots & a_{1n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{i1}+a^{'}_{i1} & a_{i2}+a^{'}_{i2} & \cdots & a_{in}+a^{'}_{in} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{n1} & a_{n2} & \cdots & a_{nn}
    \end{vmatrix}\\
    =\begin{vmatrix}
    a_{11} & a_{12} & \cdots & a_{1n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{i1} & a_{i2} & \cdots & a_{in} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{n1} & a_{n2} & \cdots & a_{nn}
    \end{vmatrix} + \begin{vmatrix}
    a_{11} & a_{12} & \cdots & a_{1n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a^{'}_{i1} & a^{'}_{i2} & \cdots & a^{'}_{in} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{n1} & a_{n2} & \cdots & a_{nn}
    \end{vmatrix}
    $$  
16. 性质6 把行列式的某一行或者列的各个元素乘以同一数然后加到玲一行或者列对应的元素上去，行列式的值不变。
17. 在n阶行列式中，把$(i,j)$元$a_{ij}$所在的第i行和第j列划去后，留下来的n-1阶行列式叫做$(i,j)$元$a_{ij}$的余子式，记作$M_{ij}$；记
    $$A_{ij}=(-1)^{i+j}M_{ij}$$
    $A_{ij}$叫做$(i,j)$元$a_{ij}$的代数余子式。
18. 一个n阶行列式，如果其中第i行所有元素除$(i,j)$元$a_{ij}$外都为0，那么这个行列式等于$a_{ij}$与它的代数余子式的乘积，即
    $$D=a_{ij}A_{ij}=a_{ij}(-1)^{i+j}M_{ij}$$
19. 定理2 行列式等于它的任一行（列）的各元素与其对应的代数余子式乘积之和，即
    $$D=a_{i1}A_{i1}+a_{i2}A_{i2}+\cdots +a_{in}A_{in}, (i=1,2,\cdots , n)$$
    或
    $$D=a_{1j}A_{1j}+a_{2j}A_{2j}+\cdots +a_{nj}A_{nj}, (j=1,2,\cdots , n).$$

### 1.4 需要记住的特例

1. $D=\begin{vmatrix}
      &   &   & a_{1n} \\
    0 &   & a_{2,n-1} & a_{2n} \\
      & \vdots & \vdots & \vdots \\
    a_{n1} & \cdots & a_{n,n-1} & a_{nn}
    \end{vmatrix}=(-1)^{\frac {1}{2} n(n-1)} a_{1n} a_{2,n-1}\cdots a_{n1}=(-1)^{\frac {n(n-1)}{2}} \prod \limits_{i=1,j=n}^{n,1} a_{ij}$
2. $D=\begin{vmatrix}
      &   &   & \lambda_{1} \\
      &   & \lambda_{2} &   \\
      & \vdots &   &   \\
    \lambda_{n} &   &   &  
    \end{vmatrix}=(-1)^{\frac {n(n-1)}{2}}\lambda_{1}\lambda_{2}\cdots \lambda_{n}=(-1)^{\frac {n(n-1)}{2}} \prod \limits_{i=1}^n \lambda_i$
3. 1和2的主要思想是通过行之间的交换行列式变号这个性质来完成计算的。
4. $$\text{设}
    D=\begin{vmatrix}
    a_{11} & \cdots & a_{1k} &   &   &   \\
    \vdots &   & \vdots &   & 0 &   \\
    a_{k1} & \cdots & a_{kk} &   &   &   \\
    c_{11} & \cdots & c_{1k} & b_{11} & \cdots & b_{1n}\\
    \vdots &   & \vdots & \vdots &  & \vdots\\
    c_{n1} & \cdots & c_{nk} & b_{n1} & \cdots & b_{nn}
    \end{vmatrix}, \\
    D_1=det(a_{ij})=\begin{vmatrix}
      a_{11} & \cdots & a_{1k} \\
      \vdots &   & \vdots \\
      a_{k1} & \cdots & a_{kk}
    \end{vmatrix}, D_2=det(b_{ij})=\begin{vmatrix}
      b_{11} & \cdots & b_{1n} \\
      \vdots &   & \vdots \\
      b_{n1} & \cdots & b_{nn}
    \end{vmatrix}. \\
    \text{则：}D=D_1 D_2$$
    这个是矩阵分块的思路。注意矩阵一定是方阵。
5. $$D_{2n}=\begin{vmatrix}
    a &  &  &  &  &b \\
      & \ddots &  & &\vdots & \\
      &  & a & b &  &  \\
      &  & c & d &  &  \\
      & \vdots &  & &\ddots & \\
    c &  &  &  &  &d
   \end{vmatrix}=(ad-bc)^n$$
   通过4中的思路，通过行与行的交换变成很多2*2的小块之后再进行计算。
6. 范德蒙德（Vandermonder）行列式
   $$D=$$

## 2. 矩阵运算

### 2.1 矩阵加法

1. $\boldsymbol{A} + \boldsymbol{B} = \boldsymbol{B} + \boldsymbol{A}$
2. $(\boldsymbol{A} + \boldsymbol{B}) + \boldsymbol{C} = \boldsymbol{A} + (\boldsymbol{B}) + \boldsymbol{C})$

### 2.2 矩阵减法

1. $\boldsymbol{A} - \boldsymbol{B} = \boldsymbol{A} + \boldsymbol{B} \times (-1)$
2. $\boldsymbol{A} - \boldsymbol{A} = \boldsymbol{O}$

### 2.3 矩阵乘法

在矩阵之间的乘法时，默认矩阵之间的行和列是满足矩阵乘法的要求的。

1. $(\lambda \mu)\boldsymbol{A} = \lambda (\mu \boldsymbol{A})$
2. $(\lambda + \mu)\boldsymbol{A} = \lambda \boldsymbol{A} + \mu \boldsymbol{A}$
3. $\lambda(\boldsymbol{A} + \boldsymbol{B}) = \lambda \boldsymbol{A} + \lambda \boldsymbol{B}$
4. $(\boldsymbol{A} \boldsymbol{B})\boldsymbol{C} = \boldsymbol{A} (\boldsymbol{B} \boldsymbol{C})$
5. $\lambda(\boldsymbol{A} \boldsymbol{B}) = (\lambda \boldsymbol{A}) \boldsymbol{B} = \boldsymbol{A} (\lambda \boldsymbol{B})$
6. $\boldsymbol{A} (\boldsymbol{B} + \boldsymbol{C}) = \boldsymbol{A} \boldsymbol{B} + \boldsymbol{A} \boldsymbol{C}\\
(\boldsymbol{B} + \boldsymbol{C}) \boldsymbol{A}  = \boldsymbol{B} \boldsymbol{A} + \boldsymbol{C} \boldsymbol{A}$

### 2.4 矩阵的转置

1. $(\boldsymbol{A}^T)^T = \boldsymbol{A}$
2. $(\boldsymbol{A} + \boldsymbol{B})^T = \boldsymbol{A}^T + \boldsymbol{B}^T$
3. $(\lambda \boldsymbol{A})^T = \lambda \boldsymbol{A}^T$
4. $(\boldsymbol{A} \boldsymbol{B})^T = \boldsymbol{B}^T \boldsymbol{A}^T$

### 2.5 矩阵的逆

下面不专门强调，在做逆运算的时候默认认为运算对象矩阵是可逆的。

1. 条件：能求逆的矩阵必须是方阵。
2. $|\boldsymbol{A}| \not ={0} \Leftrightarrow \boldsymbol{A}\text{可逆。}$
3. $\boldsymbol{A} \boldsymbol{B} = \boldsymbol{B} \boldsymbol{A} = \boldsymbol{E}$则称$\boldsymbol{A}$可逆，$\boldsymbol{A}$的逆矩阵就是$\boldsymbol{B}$。记为$\boldsymbol{A}^{-1}$。
4. $\boldsymbol{A}^{-1}$唯一。
5. 如果$\boldsymbol{A}$可逆，那么$\boldsymbol{A}^{T}$也可逆。且$(\boldsymbol{A}^{T})^{-1} = (\boldsymbol{A}^{-1})^{T}$。
6. $(\boldsymbol{A}^{-1})^{-1} = \boldsymbol{A}$
7. $k \not ={0}, (k \boldsymbol{A})^{-1} = \frac{1}{k}\boldsymbol{A}^{-1}$
8. $|\boldsymbol{A}^{-1}| = \frac{1}{|\boldsymbol{A}|}$
9. $\boldsymbol{A}^{-1} = \frac{1}{|\boldsymbol{A}|} \boldsymbol{A}^{*}$
10. 可逆矩阵乘以可逆矩阵，结果依然可逆。可逆矩阵加上可逆矩阵，结果不一定可逆。
11. 关于求逆的几个常用性质
    |1|2|3|
    |---|---|---|
    |$(\boldsymbol{A}^{-1})^{*}=(\boldsymbol{A}^{*})^{-1}$|$(\boldsymbol{A}^{-1})^T=(\boldsymbol{A}^{T})^{-1}$|$(\boldsymbol{A}^{*})^T=(\boldsymbol{A}^{T})^{*}$|
    |$(\boldsymbol{A}\boldsymbol{B})^{T}=\boldsymbol{B}^{T}\boldsymbol{A}^{T}$|$(\boldsymbol{A}\boldsymbol{B})^{*}=\boldsymbol{B}^{*}\boldsymbol{A}^{*}$|$(\boldsymbol{A}\boldsymbol{B})^{-1}=\boldsymbol{B}^{-1}\boldsymbol{A}^{-1}$|

## 范数

1. 定义。**非正式地说，一个向量的范数告诉我们一个向量有多大**。当然，定义大小有多种表达形式。在线性代数中，向量范数是将向量映射到标量的函数$f$。 给定任意向量$\boldsymbol{x}$，向量范数要满足一些属性。范数性质如下：
   1. 绝对值缩放。如果我们按常数因子$\alpha$缩放向量的所有元素， 其范数也会按相同常数因子的绝对值缩放：：$f(\alpha x) = |\alpha|f(x)$
   2. 三角不等式：$f(x+y)\leqslant f(x)+f(y)$
   3. 非负性：$f(x)\geqslant 0$
2. $L_2$范数定义。设n维向量$\boldsymbol{x}$中的元素分别是$x_1,x_2, \cdots , x_n$，那么定义$L_2$为$||x||_2 = \sqrt{\sum\limits_{i=1}^n x_i^2}$。
3. $L_1$范数定义。设n维向量$\boldsymbol{x}$中的元素分别是$x_1,x_2, \cdots , x_n$，那么定义$L_1$为$||x||_1 = \sum\limits_{i=1}^n |x_i|$。
4. $L_p$范数定义。设n维向量$\boldsymbol{x}$中的元素分别是$x_1,x_2, \cdots , x_n$，那么定义$L_p$为$||x||_p = (\sum\limits_{i=1}^n |x_i|^p)^{\frac{1}{p}}$。
5. Frobenius范数。之前的范数都是基于向量的定义。可以类比于对于矩阵而言的$L_2$范数。矩阵$\boldsymbol{X} \in \mathbb{R}^{m \times n}$的Frobenius范数定义为：$||\boldsymbol{X}||_F = \sqrt{\sum\limits_{i=0}^m\sum\limits_{j=0}^n x_{ij}^2}$。

## 重要知识点

1. 实对称矩阵就是关于主对角线的所有元素一一对称。也就是$\boldsymbol{A} = \boldsymbol{A}^T$。
   1. 要证明一个矩阵是实对称矩阵，就是需要证明$(\boldsymbol{A} \boldsymbol{A}^T)^T = \boldsymbol{A} \boldsymbol{A}^T$即可。

