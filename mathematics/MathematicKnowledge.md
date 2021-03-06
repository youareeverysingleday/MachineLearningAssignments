# Knowledge of Mathematics

You are every single day.
一般情况下求函数的极值都是使用的同一个思路：通过求驻点来找极值；如果碰到有条件限制，那么使用拉格朗日乘子法，来解决。如果函数不方便找到解析解，那么通过求梯度来找来找到收敛方向和收敛步长来找极值点。

## 常用公式

### 1 常用函数

|名称|公式|
|---|---|
|换底公式|$\log_a x = \frac{\log_b x}{\log_b a}$|
|和差|$\log xy = \log x + \log y$|

### 2 三角公式

|名称|公式|
|---|---|
|加法公式|$\sin (\alpha + \beta) = \sin \alpha \cos \beta + \cos \alpha \sin\beta$|

## 几种距离的定义

1. 参考<https://blog.csdn.net/qq_36607894/article/details/109168401>
2. L2范数和欧式距离的关系参考<https://www.cnblogs.com/lhfhaifeng/p/10671349.html>

## 最小二乘法

### 0 定义及说明

1. 参考<https://www.cnblogs.com/lhfhaifeng/p/10671349.html>
2. 以下定义是一样的：欧式距离=欧式长度=L2范数=L2距离。
3. 定义：
   $$
   \begin{aligned}
   & \text{set :} \vec{a} = [4,3,1,3,5],\; \vec{b}=[3,2,4,3,1] \\
   & \text{L2 范数可以记为以下几种形式：}||\vec{a}||, |\vec{a}|, |\vec{a}|_2 \\
   & |\vec{a}|_2 = \sqrt{\sum\limits_i^n (a_i)^2} = \sqrt{a_1^2 + a_2^2 + \cdots + a_n^2} ,\; \text{n is dimension of vector a.}\\
   & |\vec{a}|_2 = \sqrt{\sum\limits_i^n (a_i)^2} = \sqrt{4^2 + 3^2 + 1^2 + 3^2 + 5^2} = \sqrt{60}\\
   & \text{对于两个向量$\vec{a}, \vec{b}$而言，它们的L2范数可以认为是两个点的距离。}\\
   & |\vec{a}, \vec{b}|_2 = \sqrt{\sum\limits_i^n (a_i - b_i)^2} = \sqrt{(a_1-b_1)^2 + (a_2-b_2)^2 + \cdots + (a_n-b_n)^2}\\
   & |\vec{a}, \vec{b}|_2 = \sqrt{(4-3)^2 + (3-2)^2 + (1-4)^2 + (3-3)^2 + (5-1)^2} = 5\\
   \end{aligned}
   $$

### 1 最小二乘法的局限性

1. 函数形式的选取。一般的方法是通过画图之后凭借人工经验观察来设置函数形式。
2. 在选定了函数形式之后，如何设置函数的参数也是一个非常困难的地方。
   1. 设置有些参数是可以化简的，这样只是导致了增加解方程的个数，但实际上并没有起到好的效果。这个时候就需要仔细研究函数的形式。比如，$y=e^{ax+b}$设置成$y=ce^{ax+b}$，实质上后者可以化简为$y=e^{ax+b+\ln{c}}$。$b+\ln{c}$完全可以视为一个整体，并没有增加函数的自由度。
3. 超越方程组的求解。如果拟合函数不是多项式函数，也不是线性函数，超越方程组是很难解的。
   1. [神经网络和最小二乘是等价的](https://www.bilibili.com/video/BV1q741177US?from=search&seid=966740903727178826&spm_id_from=333.337.0.0)。这个视频的29:37处说出了这句话。神经网络的本质就是最小二乘法。
   2. **首先神经网络使用阶梯函数解决了函数型的选取**。用很多的阶梯函数（也就是激活函数）来拟合曲线。每个阶梯函数一般都会产生3个参数：1. 什么时候上升或者下降；2. 从多少开始变化； 3. 到多少变化结束。视频的31:00开始说这个问题。
   3. 如果分的非常细，也就是用非常多的阶梯函数去拟合，那么就会产生参数爆炸。通过增加参数的代价来解决了如何选取函数形式的问题。
   4. 用梯度下降的方法去求取超越方差组的数值解。三种拟合方法![ThreeFittingMathematicMethods](../pictures/ThreeFittingMathematicMethods.png)。该图来源于[数学建模之数据拟合（3）：最小二乘法@遇见数学](https://www.bilibili.com/video/BV1q741177US?from=search&seid=966740903727178826&spm_id_from=333.337.0.0)。

### 2 线性最小二乘法

总结：最小二乘法就是求的距离。**L2范数是表示一个向量的大小和最小二乘法的公式表示有差别**。差别在于平方之中的计算符号是正号还是负号。\
当使用最小二乘法的含义：最小，就是希望求距离最小。二乘，就是对两个变量的差值求了平方。

1. 假设在二维平面上有3个坐标点$\{x_1, y_1\}, \{x_2, y_2\}, \{x_3, y_3\}$，期望使用一个条直线去拟合。
2. 这条直线或者线性关系设为：
   $$y=ax+b \tag{1}$$
3. 评估拟合的情况需要使用评估函数，直观的评估函数的为$L(a,b)=|y_1-f(x_1)| +|y_2-f(x_2)| +|y_3-f(x_3)|$。但是存在绝对值的函数不容易使用数学工具计算（**只能使用零点分区间发来计算含有绝对值的函数。同时含有绝对值的函数不是光滑的，存在尖点；也就是说无法对其求导，所以需要变化这个评估函数**）。所以将评估函数修改
   $$L(a,b)=(y_1-f(x_1))^2 +(y_2-f(x_2))^2 +(y_3-f(x_3))^2 \tag{2} $$
   $$L(a,b)= \sum \limits_{i=1}^{n}(y_i-f(x_i))^2, \text{n为样本个数} \tag{3}$$
   **二次幂函数是光滑的，可以求高阶导数来判断它的性质**。
   1. 注意(2)中的$\boldsymbol{x}, \boldsymbol{y}$都是已有的已知量，未知量是$a,b$，$a,b$也是自变量。
4. 然后对公式(3)中所有的自变量（这里的自变量是a和b）求偏导数，并且使得偏导数为0的解即为最小值解。偏导数为0的点就是驻点。
5. 对于非线性的函数可以将函数进行线性化之后再使用最小二乘法来处理。
6. 最小二乘法的几何意义就是寻找n维空间上点的距离最小。

### 3 非线性最小二乘法

#### 3.1 参考

1. [高斯牛顿法讲解和代码实现](https://www.bilibili.com/video/BV1zE41177WB?from=search&seid=13608592245711698094&spm_id_from=333.337.0.0)
2. [牛顿法](https://www.bilibili.com/video/BV1JT4y1c7wS/?spm_id_from=autoNext)还没有看。

#### 3.2 基础知识

1. 雅可比矩阵 Jacobian Matrix
   1. 雅可比矩阵的定义。
   $$\begin{aligned}
       & \text{设}\boldsymbol{x}:[x_1, x_2, \cdots, x_n],\boldsymbol{f}:[f_1(\boldsymbol{x}),f_2(\boldsymbol{x}),\cdots,f_m(\boldsymbol{x})],\\
       & \text{雅可比矩阵是}\boldsymbol{f}对\boldsymbol{x}\text{求一阶导数，形势如下：}\\
       & \boldsymbol{J}=[\frac{\partial{\boldsymbol{f}}}{\partial{x_1}}, \frac{\partial{\boldsymbol{f}}}{\partial{x_2}},\cdots,\frac{\partial{\boldsymbol{f}}}{\partial{x_n}}]\\
       & = \begin{vmatrix}
           \frac{\partial{f_1}}{\partial{x_1}} & \cdots &\frac{\partial{f_1}}{\partial{x_n}}\\
           \frac{\partial{f_2}}{\partial{x_1}} & \cdots &\frac{\partial{f_2}}{\partial{x_n}}\\
           \vdots & \ddots &\vdots \\
           \frac{\partial{f_m}}{\partial{x_1}} & \cdots &\frac{\partial{f_m}}{\partial{x_n}}\\
       \end{vmatrix}_{m\times n}\\
   \end{aligned}$$
   2. 泰勒展开在$\boldsymbol{x_0}$处的一阶近似：$\boldsymbol{f}(\boldsymbol{x})=\boldsymbol{f}(\boldsymbol{x_0})+\boldsymbol{J}(\boldsymbol{x}-\boldsymbol{x_0})+o(||\boldsymbol{x}-\boldsymbol{x_0}||)$。
   3. 海森矩阵就是梯度的雅可比矩阵：$\boldsymbol{H}(\boldsymbol{f}(\boldsymbol{x}))=\boldsymbol{J}(\nabla\boldsymbol{f}(\boldsymbol{x}))$。

## 组合数学

1. 鸽笼原理
   1. 只能对存在性进行证明。
   2. 需要构造合适的鸽子和笼子。

## 其他

1. 图灵的停机问题说明了计算机程序不可能完成所有的功能。说明了希尔伯特关于数学完备性、一致性、可判定性三个不可能同时在一个计算机程序上完成。
2. 到目前为止，可计算能力上没有超越图灵机。其他的只是被证明与图灵机等价。

## 帕斯卡三角形

1. 性质：每行之和为$2^n$
2. 帕斯卡矩阵可逆，因为它的行列式为1.
3. 它的逆：

## 二项式定理

1. $(x+y)^n = \sum\limits_{k=0}^n C_n^k x^k y^{n-k}$
2. 证明方法：
   1. 组合分析法：思路：由$(x+y)$选择k个x，那么就剩n-k了y，那么就可以确定$x^k y^{n-k}$之前的系数了。
   2. 归纳法。
3. 代数学二项式定理
   1. $(1+x)^n = \sum\limits_{k=0}^n C_n^k x^k$
   2. 和泰勒展开非常像。
4. $1C_n^1 + 2C_n^2 + \cdots +nC_n^n = n\cdot 2^{n-1}$
   1. 对 $(1+x)^n = \sum\limits_{k=0}^n C_n^k x^k$两边求导
   2. 然后将x=1带入即可证明。
   3. 组合意义：在n个人中选择k个人做做主席团，然后从k个人中选一个做主席。两种解释：一种是先选主席，再选委员；第二种是先选委员，再从委员中选主席。
5. 范德蒙等式$\sum\limits_{k=0}^n (C_n^k)^2 = C_{2n}^n$
6. $\sum\limits_{i=0}^r C_m^i C_{r-i}^n = C_{m+n}^r$
   1. 组合分析证明：左边表示的从m+n个中选r个.右边表示先从m个中选i个，从n个中选择r-i个，然后i再从0取值到r。
7. 非降路径法：证明组合恒等式的效果非常好。
   1. 需要理解的地方：终点坐标的确定。
   2. 按照什么方式进行分类来确定等号复杂一边的意义。
   3. 适用场景是等号一边非常简单。
8. 二项式系数的单峰性：二项式系数的大小在中间的时候最大，而且两边是对称的。
   1. 关系就是一个集合。
   2. 偏序集基本概念
   3. Sperner定理
      1. 最大链
      2. 反链