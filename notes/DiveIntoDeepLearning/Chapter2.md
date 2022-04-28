# preview knowledge

## 2.1 数据操作

[tensorflow基础操作](../../codes/8TensorFlowGuide/1TensorFlowFoundation.ipynb)

## 2.2 数据预处理

[Pandas手册](../../codes/7PandasGuide/0PandasGuide.ipynb)

## 2.3 线性代数

[tensorflow基础操作](../../codes/8TensorFlowGuide/1TensorFlowFoundation.ipynb)
[线性代数](../../mathematics/LinearAlgebra.md)

## 2.4 微积分

优化（optimization）:用模型拟合观测数据的过程；
泛化（generalization）：数学原理和实践者的智慧，能够指导我们生成出有效性超出用于训练的数据集本身的模型。

## 2.5 自动微分

详见[Tensorflow foundation](../../codes/8TensorFlowGuide/1TensorFlowFoundation.ipynb求导部分的代码。

2.5.2 非标量变量的反向传播

    关于2.5.2“非标量变量的反向传播”中这部分内容没明白要表达的具体含义。

    当y不是标量时，向量y关于向量x的导数的最自然解释是一个矩阵。 对于高阶和高维的y和x，求导的结果可以是一个高阶张量。
    然而，虽然这些更奇特的对象确实出现在高级机器学习中（包括深度学习中）， 但当我们调用向量的反向计算时，我们通常会试图计算一批训练样本中每个组成部分的损失函数的导数。 这里，我们的目的不是计算微分矩阵，而是单独计算批量中每个样本的偏导数之和。

2.5.3 分离计算
    这里应该是实现了链式法则。

    有时，我们希望将某些计算移动到记录的计算图之外。 例如，假设y是作为x的函数计算的，而z则是作为y和x的函数计算的。 想象一下，我们想计算z关于x的梯度，但由于某种原因，我们希望将y视为一个常数， 并且只考虑到x在y被计算后发挥的作用。
    在这里，我们可以分离y来返回一个新变量u，该变量与y具有相同的值， 但丢弃计算图中如何计算y的任何信息。 换句话说，梯度不会向后流经u到x。 因此，下面的反向传播函数计算z=u*x关于x的偏导数，同时将u作为常数处理， 而不是z=x*x*x关于x的偏导数。

## 2.6 概率论

1. 代码详见[tensorflow_probability](../../codes/8TensorFlowGuide/3TensorFlowProbability.ipynb部分的代码。本书：“**我们将考虑离散空间中的概率**”（并不考虑连续空间的概率，因为没有意义）。
2. [理论部分详见](../../mathematics/ProbabilityTheory.md)。