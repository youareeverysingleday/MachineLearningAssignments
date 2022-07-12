# causal inference 因果推理

参考：

1. [在课程上有同学分享了因果的论文，这个是它看的视频](https://www.bilibili.com/video/BV1jF411M7zu/)

机器学习的本质是什么？或者说学习的本质是什么？
学习是学习已有的知识来对未知的世界（事件）进行预判。也就是举一反三的能力。
实现人工智能的三个流派：符号主义、连接主义和行为主义。

## 1. 背景知识介绍：为什么需要因果

几种现象：

1. 隐混淆（辛普森悖论）
2. 选择偏差

发展方向：

1. 因果知识从哪里来？、
   1. 通过因果发现的方法去进行因果推理和反事实推理。也就是攀爬因果之梯。
2. 因果知识怎么利用？
   1. 因果性学习，**将因果关系和深度学习方法相结合，替换原来是基于相关性学习的方法**。从而实现强泛化和可解释的人工智能。

## 2. 因果发现

### 2.1 什么是因果发现

1. 实际上是回答一个现象为什么会发生的问题。为了回答这问题一般会需要3个要素：观察、假设、实验。
2. 基于干预实验的方法
   1. 核心思想：干预原因，观察结果会发生什么变化。
   2. 缺陷：很多实验的代价太大，并不适用绝大多数情况。
3. 基于观察数据的方法
   1. 核心思想：观察数据+因果假设$\Rightarrow$因果模型。
   2. 这种中间的关键点在于“因果假设”。如果没有因果假设，我们在观察数据中发现因果关系的。这也是统计与因果的不同之处就在于这里。
   3. 因果假设不可论证和检验的。
   4. 如果要判断一个方法是因果方法还是统计方法，就看这个方法是否可以验证。如果能够验证肯定就是统计方法，如果不能够验证，那么就是因果方法。
   5. **因果假设只能基于先验知识来做（这里应该是验证的意思）**。可以验证的肯定是基于统计的方法。

### 2.2 经典方法：基于约束的方法、基于因果函数的方法

经典因果模型：structural causal models and graphical causal models

#### 2.2.1 基于约束的因果发现方法

#### 2.2.2 基于函数的因果发现方法

CANM模型
PNL模型

#### 2.2.3 混合型因果发现方法

### 2.3 研究进展：隐变量问题、非独立同分布问题

### 2.4 应用探索：故障检测

## 3. 因果性学习

## 4. 基础

[参考来源](https://mp.weixin.qq.com/s/iPzfrQi6tWHckdm92ACN9g)

本文从以下六个方面来阐述，介绍内容较基础。

基本概念
难题和挑战
经典因果推断模型
子空间因果推断模型
深度表征学习因果推断模型
参考文献

一、基本概念

因果关系Causality指的是Cause和Effect，在很多领域被广泛应用，例如数据分析, 哲学、心理学、经济学、教育和医学等。

Causation和Correlation的区别，因果关系的存在，必然会伴随着相关性。但是，从因到果还需时间上的先后顺序、以及合理的机制等。因此，相关性只是因果关系的必要不充分条件。相关性并不一定代表着有因果关系。

因果推断和因果发现，因果发现是是通过计算方法从大量数据中识别因果关系，因果推理是根据结果发生的条件对因果关系作出结论的过程。

Experimental Study和Observational Study，实验学习中样本是随机的，treatment group和control group都是随机分配的，但是Observational Study中Treatment的分配一定是有策略的，非随机的。

因果推断有两个经典框架，一个是基于Judea Pearl的结构因果模型Structure Causal Model，一个是基于Rubin提出的Potential Outcome Framework。不过二者在底层原理上也是相同的。下面重点介绍一些Rubin的POF框架中的相关概念。

1. Unit，研究对象
2. Treatment，施加在研究对象上的Action
3. Outcome，在Unit被施加Treatment/Control后的输出结果
4. Treatment Effect，当施加不同的Treatment时，Unit的Outcome的变化
5. Potential Outcome，Unit被施加Treatment后所有可能的输出被称为潜在结果
6. Observed Outcome，实验观测到的Unit被施加Treatment时的输出结果
7. Counterfactual Outcome，实验中Unit没有发生的潜在结果，称为反事实结果

Treatment Effect的评估指标：ATE，ITE和CATE

1. ATE，Average Treatment Effect，人群级别的评估指标，计算方式如下
2. ITE，Individual Treatment Effect，个体级别的评估指标，计算方式如下
3. CATE，Conditional Average Treatment Effect，Subgroup级别的评估指标，计算方式如下

三大重要假设

1. Stable Unit Treatment Value Assumption SUTVA：Unit之间是相互独立的，即当对一个Unit施加treatment之后，不会影响其他Unit的Outcome；
2. Ignorability：在给定X的情况下，Treatment和Potential Outcome之间是相互独立的；
3. Positivity：对于任意一组X的值，Treatment是不确定的，即X和Treatment是随机的。

二、难题和挑战
重要概念：Confounders

Confounder是指实验中的一种变量，同时影响了Treatment，又影响了outcome，当实验中农存在这种变量时，便可能出现辛普森悖论。

上述表格中Age便是一个Confounder，age同时影响了治疗方式Treatment，又影响治疗效果，所以分组数据Young和Older，结论都是Treatment B的治疗效果更好，但是总体数据却得出Treatment A的治疗效果更好，这就是辛普森悖论。

重要概念：Selection Bias

Selection Bias是指观测组的数据分布不具有代表性，直白点来说就是X和Treatment之间不是相互独立的，会存在偏差。Confounder变量的存在会影响Unit对于Treatment的选择，进而导致了selection bias，进而selection bias又会使得counterfactual outcome的预估变得更加困难。

结论：Confounder好Selection bias是Causal inference中的两个重大难题，很多方法都是在着力解决这两大问题。

三、经典因果推断模型

1. Re-weighting methods，核心思想：为了解决数据中存在的selection bias，通过给观察数据集中的每个样本分配适当的权重，建立了一个伪总体，在这个伪总体上实验组和对照组的分布是相似的，权重的计算通过propensity-score methods来求解。

2. Mathching methods，核心思想：通过距离函数计算，将相似的数据分别分到实验组和对照组，该方法在估计反事实的同时，减少了由混杂因素带来的估计偏差。使用较多的matching方式时propensity score matching。

3. Tree-based methods，核心思想：是一种基于决策树的预测模型，如分类树和回归树。在CART中，一棵树被建立直到达到分裂容忍。这里只有一棵树，可以根据需要进行生长和修剪

4. Stratification methods，核心思想：也是为了解决数据中存在的selection bias，通过将整个组分成子组来调整选择偏倚，在每个子组中，处理组和对照组在某些测量下是相似的

5. Multitask Learning methods，实验组和对照组使用不同的模型，共享一些共同的特点

6. Meta-Learning methods，例如：S-learner，T-Learner，X-learner，R-learner等，是一个系列的解决方案。


四、子空间因果推断模型
核心思想：在original data space中执行matching是简单方便的，但是缺点是容易被不影响outcome的变量所误导，因此，可以映射到subspace进行matching来解决该问题。存在的方案有

1. NNM with Random Subspaces
2. Informative Subspace Learning
3. Nonlinear and Balanced Subspace Learning
这里给大家分享几篇经典论文

2016 Large sample properties of matching estimators for average treatment effects.
AAAI 2017 Informative Subspace Learning for Counterfactual Inference
IJCAI 2016 Matching via Dimensionality Reduction for Estimation of Treatment Effects in Digital Marketing Campaigns
NIPS 2017 Matching on balanced nonlinear representations for treatment effects estimation

五、深度表征学习因果推断模型
该方法将因果推断和深度学习相结合，分为以下三类

1. Balanced representation learning
2. Local similarity preserving based methods
3. Deep generative model based methods
这里给大家推荐几篇经典论文

ICDM 2019 ACE- Adaptively Similarity-Preserved Representation Learning for Individual Treatment Effect Estimation.
ICLR 2018 GANITE Estimation of Individualized Treatment Effects using Generative Adversarial Nets
ICML 2016 Learning Representations for Counterfactual Inference
IJCAI 2019 On the estimation of treatment effect with text covariates.
JMLR 2017 Estimating individual treatment effect- generalization bounds and algorithms
NIPS 2017 Causal Effect Inference with Deep Latent-Variable Models
NIPS 2018  Representation learning for treatment effect estimation from observational data


六、参考文献
A Survey on Causal Inference
Causal Inference in Machine Learning
Machine Learning for Causal Inference
From how to why: An overview of causal inference in machine learning