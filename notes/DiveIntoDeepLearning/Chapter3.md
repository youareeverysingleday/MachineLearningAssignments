# 线性神经网络  linear neural network

## 3.1 线性回归 linear regression

1. 放射变换（affline transformation）。线性回归假设的模型是输入特征的一个仿射变换。仿射变换的特点是通过加权和对特征的进行线性变化，并通过偏置项来进行平移得到的。
2. 通常用$\hat{y}$来表示估计值。
3. 虽然我们相信给定$\boldsymbol{x}$预测的最佳模型会是线性的， 但我们很难找到一个有个$n$样本的真实数据集，其中对于所有的$1\leqslant i \leqslant y$, $y^{(i)}$完全等于$\boldsymbol{w}^T\boldsymbol{x}^{(i)}+\boldsymbol{b}$。无论我们使用什么手段来观察特征$\boldsymbol{X}$和标签$\boldsymbol{y}$，**都可能会出现少量的观测误差**。因此，即使确信特征与标签的潜在关系是线性的，我们也会加入一个噪声项来考虑观测误差带来的影响。
4. 由于平方误差函数中的二次方项，估计值$\hat{y}^{(i)}$和观测值$y^{(i)}$之间较大的差异将导致更大的损失。为了度量模型在整个数据集上的质量，我们需计算在训练集个样本上的损失均值（也等价于求和）。
    $$L(\boldsymbol{w}, \boldsymbol{b})=\frac{1}{n}\sum\limits_{i=1}^n l^{(i)}(\boldsymbol{w},\boldsymbol{b})=\frac{1}{n}\sum\limits_{i=1}^n \frac{1}{2} (\boldsymbol{w}^{T}\boldsymbol{x}^{(i)}+\boldsymbol{b}-\boldsymbol{y}^{(i)})^2 \tag{3.1.6}$$
    在训练模型时，我们希望寻找一组参数$(\boldsymbol{w}^*, \boldsymbol{b}^*)$，这组参数能最小化在所有训练样本上的总损失。如下式：$\boldsymbol{w}^*, \boldsymbol{b}^* =\underset{\boldsymbol{w},\boldsymbol{b}}{argmin}L(\boldsymbol{w},\boldsymbol{b})$。
5. 解析解：$\boldsymbol{w}^* =(\boldsymbol{X}^T \boldsymbol{X})^{-1} \boldsymbol{X}^T \boldsymbol{y}$。像线性回归这样的简单问题存在解析解，但并不是所有的问题都存在解析解。解析解可以进行很好的数学分析，但解析解对问题的限制很严格，导致它无法广泛应用在深度学习里。[解答参考文字](https://zhuanlan.zhihu.com/p/74157986)。[解答参考视频](https://www.bilibili.com/video/BV1ro4y1k7YA?spm_id_from=333.337.search-card.all.click)。
   1. 这里如何得来的？
   2. 为什么还有$\boldsymbol{X}^T$这一项？答：这是为了表示他们之间的距离（L2范数）。**两个形状相同向量之间的距离可以表示为一个向量的转置乘以另外一个向量**。推导过程如下：

        $$
        \begin{cases}
        \text{Known:} & \text{the sample space is}\{(x_1,y_1),(x_2,y_2),\cdots (x_n,y_n)\}\\
         & \text{equation: }\boldsymbol{Y}=\boldsymbol{X}\boldsymbol{B}\\
         & \boldsymbol{Y}=\begin{pmatrix}
            y_1 \\
            y_2 \\
            \vdots \\
            y_n
            \end{pmatrix};
        \boldsymbol{X}=\begin{bmatrix}
            1 & x_1 \\
            1 & x_2 \\
            & \vdots \\
            1 & x_n
            \end{bmatrix};
        \boldsymbol{B}=\begin{pmatrix}
            \alpha \\
            \beta
            \end{pmatrix}.\\
         & \alpha\text{ is intercept,} \beta \text{ is slope.} \\
         & \therefore \boldsymbol{Y} = \boldsymbol{X}\boldsymbol{B} + \boldsymbol{\gamma} \\
         & \boldsymbol{\gamma} =\begin{pmatrix}
            \epsilon_1 \\
            \epsilon_2 \\
            \vdots \\
            \epsilon_n
            \end{pmatrix};\\

        \text{Target is: } & \boldsymbol{w}^*, \boldsymbol{b}^* =\underset{\boldsymbol{w},\boldsymbol{b}}{argmin}L(\boldsymbol{w},\boldsymbol{b})\\
        \text{solution: } & \\
         & \boldsymbol{X}\boldsymbol{B} - \vec{\boldsymbol{y}} = \begin{bmatrix}
            \vec{x}_1^T b \\
            \vec{x}_2^T b \\
            \vdots \\
            \vec{x}_n^T b
            \end{bmatrix} - \begin{bmatrix}
            \vec{y}_1 \\
            \vec{y}_2 \\
            \vdots \\
            \vec{y}_n
        \end{bmatrix}\\
        & = \begin{bmatrix}
            \vec{x}_1^T b - \vec{y}_1\\
            \vec{x}_2^T b - \vec{y}_2\\
            \vdots \\
            \vec{x}_n^T b - \vec{y}_n
        \end{bmatrix}\\
         & \because \vec{z}^T\vec{z} = \sum_i z_i^2;\\
         & \therefore \begin{bmatrix}
            \vec{x}_1^T b - \vec{y}_1\\
            \vec{x}_2^T b - \vec{y}_2\\
            \vdots \\
            \vec{x}_n^T b - \vec{y}_n
            \end{bmatrix} \\
         & = \frac{1}{2} (\boldsymbol{X}\boldsymbol{B} - \vec{\boldsymbol{y}})^T (\boldsymbol{X}\boldsymbol{B} - \vec{\boldsymbol{y}}) = \frac{1}{2} \sum\limits_{i=1}^n(\vec{x}_i^T b - \vec{y}_i)\\
         & =L(\boldsymbol{B})\\
         & \nabla_B L(\boldsymbol{B}) = \nabla_B \frac{1}{2} (\boldsymbol{X}\boldsymbol{B} - \vec{\boldsymbol{y}})^T (\boldsymbol{X}\boldsymbol{B} - \vec{\boldsymbol{y}}) \\
         & = \frac{1}{2} \nabla_B ((\boldsymbol{X}\boldsymbol{B})^T \boldsymbol{X}\boldsymbol{B} - (\boldsymbol{X}\boldsymbol{B})^T \vec{y} - \vec{y}^T(\boldsymbol{X}\boldsymbol{B}) + \vec{y}^T \vec{y}) \\
         & \because \vec{a}^T\vec{b} = \vec{b}^T\vec{a} \;\text{ and }\;\vec{y}^T \vec{y}\text{ is independent of }\boldsymbol{B}.\\
         & \because \nabla_x \vec{b}^T x =  \vec{b}\;\text{ and }\;\nabla_x \vec{x}^T \boldsymbol{A}x =  2\boldsymbol{A}\vec{x}\;\text{ for sysmmetric matrix.}\\
         & \therefore = \frac{1}{2} \nabla_B (\boldsymbol{B}^T (\boldsymbol{X}^T \boldsymbol{X})\boldsymbol{B} - (\boldsymbol{X}\boldsymbol{B})^T \vec{y} - \vec{y}^T(\boldsymbol{X}\boldsymbol{B})) \\
         & = \frac{1}{2} \nabla_B (\boldsymbol{B}^T (\boldsymbol{X}^T \boldsymbol{X})\boldsymbol{B} - \vec{y}^T(\boldsymbol{X}\boldsymbol{B})  - \vec{y}^T(\boldsymbol{X}\boldsymbol{B})) \\
         & = \frac{1}{2}(2(\boldsymbol{X}^T \boldsymbol{X})\boldsymbol{B}-2\vec{y}^T \boldsymbol{X})\\
         & = (\boldsymbol{X}^T \boldsymbol{X})\boldsymbol{B}-\vec{y}^T \boldsymbol{X}\\
         & \text{To minimize L, L is convex function, we set its derivatives to zero, and obtain the normal equations:}\\
         & (\boldsymbol{X}^T \boldsymbol{X})\boldsymbol{B} = \vec{y}^T\boldsymbol{X} = \boldsymbol{X}^T\vec{y}\\
         & \Rightarrow \boldsymbol{B} = (\boldsymbol{X}^T \boldsymbol{X})^{-1}\boldsymbol{X}^T \vec{y} \\
         & \text{Proof complete.}
        \end{cases}
        $$

