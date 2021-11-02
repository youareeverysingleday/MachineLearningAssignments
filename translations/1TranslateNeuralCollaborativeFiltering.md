# translate翻译

翻译说明：
|number|title of paper|internet source|local source|correlative field|illustration|
|---|---|---|---|---|---|
|1|neural collaborative filtering|http://staff.ustc.edu.cn/~hexn/papers/www17-ncf.pdf|./references/1NeuralCollaborativeFiltering.pdf|recommonder system|English translate into chinese|

## 1. neural collaborative filtering

1. 题目
神经协同过滤

## 2. abstract 摘要

|编号|英语|本文中翻译的中文|理解|
|---|---|---|---|
|1|immense|巨大的|/|
|2|speech recognition|语音识别|/|
|3|Neural network-based Collaborative Filtering|NCF基于协同过滤的神经网络|/|
|4|arbitrary function|任意函数|可能不是特指什么函数，而是泛指所有函数。|
|5|implicit feedback|隐式反馈|显式反馈就是用户对item的打分。例如按照评分1-5来打分，不同的打分就代表用户对item的不同喜好程度。隐式反馈不是打分，可以看做是一种选择。用户选择了某个item。举例来说这个选择可以是社交网络中的点赞、转发等等。除了用户所选择的item之外，剩下的item都是未选择的，而未选择不代表用户不喜欢这些item。有可能是因为还未看到这些item等原因。这就是隐式反馈与显示反馈的区别。例如Gowalla这样的数据，就是隐式反馈数据。[参考](https://blog.csdn.net/yinruiyang94/article/details/78906370)|
|6|state of the art|传说中的SOTA|重大改进和显著提高|
|7|matrix factorization|矩阵分解|把矩阵A分解为矩阵U和V的乘积。[参考](https://blog.csdn.net/u014595019/article/details/80586438)|
|8|empirical evidence|实验证据表明|有些地方把它翻译为“经验表明”，这里理解为实验表明。|


在最近几年，深度神经网络已经在语音识别、计算机视觉和自然语言处理领域取得的巨大的成功。然而，深度神经网络在推荐系统的探索较少的受到关注（scrutiny）。在这个工作中，我们努力开发了基于神经网络的技术来处理推荐系统中的关键问题——协同过滤——在隐式反馈（implicit feedback）的基础上。  
尽管最近有些工作已经在推荐系统中使用了深度学习，他们主要使用它去模拟辅助信息，诸如对象（item）的文字描述和音乐声纹特征。在分析用户和对象特征之间的交互作用时，虽然深度学习已经成为协同过滤模型的关键因素，它们仍然使用（resort）矩阵分解的方法，并对用户和对象的隐藏特征进行内积操作。  
通过将内积替换为一个可以从数据中学习任意函数（arbitrary function）的神经结构，我们提出了一个名为：NCF的通用框架，NCF是基于神经网络的协同过滤的简称。NCF是通用的，并且再它的框架下NCF能表达、推广（express and generalize）矩阵分解。为了增强NCF模型在非线性情况下的表现，我们计划借助（leverage）多层感知机来学习用户项(item)交互函数。在两个真实数据集上的大量实验表明，与最先进的方法相比（state of the art）我们提出的NCF框架有了显著的提高（significant improvements）。实验证据表明（empirical evidence）使用更深层的神经网络可以提供更好的推荐性能。  

关键词
协同过滤，神经网络，深度学习，矩阵分解，隐式反馈。

## 3. 介绍

|编号|英语|本文中翻译的中文|理解|
|---|---|---|---|
|1|project|映射，投影|应该表达了空间映射的关系|
|2|shared latent space|共享的潜在空间|可能是将两种数据共同映射到了一个使它们可以发生联系的空间中，是不是在暗示对**矩阵进行线性变化之后对应到一个可能线性相关的空间中了**。|
|3|represent|表示、代表|注意后面的数量形式，加了量词a和an。|
|4|inner product|内积|矩阵乘法|
|5|de facto|事实上的|/|
|6|approach|方法，名词|这里表示的是名词方法，不是动词。|
|7|model-based|模型|是一个专有名词，有对比的两个词组model-free和model-based，是两种方法。model-based可以理解为人工提取特征的，也就是特征在模型运算过程中不能更改。而对应的神经网络是“学习”特征，这种特征是潜在的、非人工指定的。[参考1](https://zhidao.baidu.com/question/694026622784498004.html)[参考2](https://www.zhihu.com/question/64369408)。**还是没有理解这个位置**。|
|8|factorization machines|因子分解机|[参考](https://blog.csdn.net/lijingru1/article/details/88623136)|
|||||

在信息爆炸的时代，推荐系统在缓解信息过载扮演着关键作用，而且它已经在众多在线服务中广泛使用，比如电子商务、在线新闻和社交网站。个性化的推荐系统关键在于根据用户过去与对象（item）之间的互动（比如评分和点击）而表现出来的偏好进行建模，成为协同过滤。在各种协同过滤技术中，矩阵分解（MF）是最流行的一种，他将用户和对象的映射到一个共享的潜在空间中，使用潜在的特征向量来代表**一个**用户和**一个**对象。以后，用户和对象（item）的交互被做成它们潜在向量的内积模型。

通过Netfix奖的推广，矩阵分解已经成为事实上方法，~~这种方法是潜在因素model-based推荐系统的~~。许多研究工作致力于增强矩阵分解，例如将其MF和基于邻居（neighbor-based）的模型集成、将MF与item内容的主题模型（topic models）组合起来，并将MF扩展到分解机（factorization machines），以便于对特征




|编号|英语|中文|
|---|---|---|
||||
||||
||||
