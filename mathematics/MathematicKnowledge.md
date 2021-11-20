# Knowledge of Mathematics

You are every single day.

## 1. 线性代数

### 1.1 行列式

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
    a_{i1}+a'_{i1} & a_{i2}+a'_{i2} & \cdots & a_{in}+a'_{in} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{n1} & a_{n2} & \cdots & a_{nn}
    \end{vmatrix}
    $$
    则D等于下列两个行列式之和：
    $$
    D=\begin{vmatrix}
    a_{11} & a_{12} & \cdots & a_{1n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{i1}+a'_{i1} & a_{i2}+a'_{i2} & \cdots & a_{in}+a'_{in} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{n1} & a_{n2} & \cdots & a_{nn}
    \end{vmatrix}
    =\begin{vmatrix}
    a_{11} & a_{12} & \cdots & a_{1n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{i1} & a_{i2} & \cdots & a_{in} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{n1} & a_{n2} & \cdots & a_{nn}
    \end{vmatrix} + \begin{vmatrix}
    a_{11} & a_{12} & \cdots & a_{1n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a'_{i1} & a'_{i2} & \cdots & a'_{in} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{n1} & a_{n2} & \cdots & a_{nn}
    \end{vmatrix}
    $$
16. 性质6 把行列式的某一行或者列的各个元素乘以同一数然后加到玲一行或者列对应的元素上去，行列式的值不变。
