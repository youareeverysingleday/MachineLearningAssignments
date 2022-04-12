# 矩阵因子分解

1. 重点-参考视频<https://www.bilibili.com/video/BV1Ah411v7Vn?p=4&spm_id_from=pageDriver>

## 1. 基本概念

1. 意义：
   1. 为了解决矩阵稀疏的问题。
   2. 隐性空间维度k。隐性空间是不可解释的。如果k越大说明模型越精细，也就是说将物品的划分的分类越具体；k越小表明模型泛化能力越强。k的大小决定了隐性空间的表达能力。
   3. 如果特征数量大于样本数量时会无法训练。一般情况下都需要选择比较小的k。
   4. 将高维矩阵分解为低维矩阵的乘积。
2. 问题定义：
   $$Preference(u,i)=R_{u,i}=P_u Q_i=\sum\limits_{f=1}^T p_{u,k}q_{k,i} \tag{1}$$

## 2. 几种方式

### 2.1. 特征值分解

1. 缺点：只能对方阵进行分解。

### 2.2. 奇异值分解

1. 定义
   1. 参考：统计学习方法第15章。
   2. 数学定义：$\boldsymbol{A} = \boldsymbol{U}_{m\times m} \boldsymbol{\Sigma}_{m \times n} \boldsymbol{V}_{n \times n}^T$，其中$\boldsymbol{A}$是$m\times n$的实矩阵。$\boldsymbol{U}$是m阶正交矩阵。$\boldsymbol{\Sigma}$是$m\times n$的矩形对角矩阵，对角元素非负且降序排列。$\boldsymbol{V}^T$是n阶正交矩阵。
   3. 对角矩阵不一定是方阵。
   4. 正交矩阵的定义：$\boldsymbol{U}\boldsymbol{U}^T= \boldsymbol{E}$。
   5. 在奇异值分解当中$\boldsymbol{\Sigma}_{m \times n}$是唯一的；但是$\boldsymbol{U}_{m\times m}$和$\boldsymbol{V}_{n \times n}^T$均不唯一。
2. 定理说明了**一个实矩阵的奇异值分解一定存在**。
3. 计算步骤：
   1. 构造实对称矩阵：$\boldsymbol{W} = \boldsymbol{A}^T \boldsymbol{A}$。实对称矩阵就是关于主对角线的所有元素一一对称。也就是$\boldsymbol{A} = \boldsymbol{A}^T$。
   2. 计算$\boldsymbol{W}$的特征值和特征向量。
      1. 求解特征方程$(\boldsymbol{W}-\lambda \boldsymbol{I})x = 0$来求得特征值$\lambda_i$，并将$\lambda_i(i=1,2,\cdots,n)$由大到小排列。然后将特征值带入特征方程求得对应每个特征值的特征向量。
   3. 利用上述求得的特征向量，求得n阶正交矩阵$\boldsymbol{V}_{n \times n}$。
      1. 将特征向量单位化，得到单位特征向量$v_1,v_2,\cdots,v_n$，构成n阶正交矩阵$\boldsymbol{V}=[v_1\;v_2\;\cdots\;v_n]$
   4. 利用上述求得的特征值，求得$m \times n$对角矩阵$\boldsymbol{\Sigma}$。
      1. 计算A的奇异值$\sigma_i=\sqrt{\lambda_i},\;i=1,2,\cdots,n$；将构造$m\times n$矩形对角矩阵$\boldsymbol{\Sigma}$，主对角线元素全是奇异值，其余元素全是0。$\boldsymbol{\Sigma}=diag(\sigma_1,\sigma_2,\cdots,\sigma_n)$。一定注意$\sigma_1,\sigma_2,\cdots,\sigma_n$也是按从大到小的顺序排列的。
   5. 利用上述求得的$\boldsymbol{V}_{n \times n}$和$\boldsymbol{\Sigma}$，求得m阶正交矩阵$\boldsymbol{U}$。分为3步。
      1. 先求$\boldsymbol{U}_1$。对$\boldsymbol{A}$的前r个正奇异值（r是由n个中的$\lambda$大于0的奇异值的个数为r），令$u_{j}=\frac{1}{\sigma_j}\boldsymbol{A}v_j,\;j=1,2,\cdots,r$得到$\boldsymbol{U}_1=[u_1\,u_2\,\cdots\,u_r]$。
      2. 再求$\boldsymbol{U}_2$。求$\boldsymbol{A}^T$的零空间的一组标准正交基$\{u_{r+1},u_{r+2},\cdots,u_{m}\}$，令$\boldsymbol{U}_2=[u_{r+1}\,u_{r+2}\,\cdots\,u_m]$。这里没有说m的个数是多少。
      3. $\boldsymbol{U}=[\boldsymbol{U}_1 \; \boldsymbol{U}_2]$。
4. 缺点：
   1. 要求原始矩阵是稠密的。对于稀疏矩阵就需要将其补全，但实际上是无法有效补全的（或者在数据层面是无法补全的）。
   2. SVD计算复杂度非常高。而我们的用户-物品矩阵非常大，所及基本上无法使用。所以SVD基本上是无法在推荐系统中使用的。

### Basic SVD(LFM, Funk SVD这三者实际上是同一种东西。将矩阵分解问题转化为了最优化问题。然后通过梯度下降来进行优化。)

1. 将矩阵分解问题转化为了最优化问题。然后通过梯度下降来进行优化。
2. 预测函数还是公式1。
3. 用误差平方和来定义损失函数。
   $$SSE = \sum\limits_{u,i}e_{ui}^2=\sum\limits_{u,i}(r_{ui}-\sum\limits_{k=1}^Kp_{u,k}q_{k,i})$$
4. 优化目标：$\underset{p,q}{min}\sum\limits_{(u,i)\in K}(r_{ui}-p_u q_i)^2$
5. 优化的方法就是梯度下降
   1. 求梯度
   2. 梯度更新