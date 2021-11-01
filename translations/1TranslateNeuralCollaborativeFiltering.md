# translate翻译

翻译说明：
|number|title of paper|internet source|local source|correlative field|illustration|
|---|---|---|---|---|---|
|1|neural collaborative filtering|http://staff.ustc.edu.cn/~hexn/papers/www17-ncf.pdf|./references/1NeuralCollaborativeFiltering.pdf|recommonder system|English translate into chinese|

## 1. neural collaborative filtering

1. 题目
神经协同过滤

## 2. abstract 摘要

在最近几年，深度神经网络已经在语音识别、计算机视觉和自然语言处理领域取得的巨大的成功。然而，深度神经网络在推荐系统的探索较少的受到关注（scrutiny）。在这个工作中，我们努力开发了基于神经网络的技术来处理推荐系统中的关键问题——协同过滤——在隐式反馈（implicit feedback）的基础上。  
尽管最近有些工作已经在推荐系统中使用了深度学习，他们主要使用它去模拟辅助信息，诸如对象（item）的文字描述和音乐声纹特征。在分析用户和对象特征之间的交互作用时，虽然深度学习已经成为协同过滤模型的关键因素，它们仍然使用（resort）矩阵分解的方法，并对用户和对象的隐藏特征进行内积操作。  
通过将内积替换为一个可以从数据中学习任意函数（arbitrary function）的神经结构，我们提出了一个名为：NCF的通用框架，NCF是基于神经网络的协同过滤的简称。NCF是通用的，并且再它的框架下NCF能表达、推广（express and generalize）矩阵分解。为了增强NCF模型在非线性情况下的表现，我们计划借助（leverage）多层感知机来学习用户项(item)交互函数。在两个真实数据集上的大量实验表明，与最先进的方法相比（state of the art）我们提出的NCF框架有了显著的提高（significant improvements）。实验证据表明（empirical evidence）使用更深层的神经网络可以提供更好的推荐性能。  

关键词
协同过滤，神经网络，深度学习，矩阵分解，隐式反馈。

## 3. 介绍


