# Speed Reading 速读

1. 题目：Effective fine-grained location prediction based on user check-in pattern in LBSNs
基于上下文感知和注意的数据增强的POI推荐

|number|title of paper|internet source|local source|correlative field|illustration|
|---|---|---|---|---|---|
|1|Learning Graph-based POI Embedding for Location-based Recommendation|<https://sci-hub.ee/10.1016/j.jnca.2018.02.007>|/|location prediction|distll|

## 问题

1. 在贡献中，训练了两个不同的模型来分类一个给定的用户是否在给定的时间下载一个候选位置check-in。为什么需要训练两个模型？

## 需要进一步了解的

1. [Hawkes process（霍克斯过程）](https://dreamhomes.top/posts/202106241018/)。简单定义：自/互激励过程（self/mutual-exciting process），亦称为霍克斯过程（Hawkes processes，以1971年提出者Hawkes教授姓氏命名），主要思想：发生的历史事件对于未来事件的发生有激励作用（正向作用），并且假设历史事件对未来的影响是单调指数递减的，然后以累加的形式进行叠加。
   1. 之前的做法是：对机器学习和其他学科来说，有趣的自然现象包括时间作为分析的中心维度。那么，一项关键任务就是捕捉和理解时间线上的统计关系。处理时态数据的主力模型是在时间序列分析下收集的。**这类模型通常将时间划分为大小相等的桶，并将数量与模型操作的每个桶相关联。这就是离散时间形式主义**，出现在许多机器学习中常见的模型中，比如卡尔曼滤波器和隐马尔可夫模型，正如计量经济学和预测学中常见的模型，如ARIMA或指数平滑法。[不好的参考，看看前面的对于其他做法的分析](https://blog.csdn.net/fs1341825137/article/details/116951405)

## 简介

1. GPS数据很稠密，但是没有语义信息。而且暴露了隐私。
2. user call records通过用户通话记录中获取位置信息。但是这种数据的精度有问题，也就是无法区分两个非常近的位置。
3. check-in数据更精确，而且有相关的一些信息。使用check-in信息可以保证解这个问题的普遍性。
4. 目前的局限性（没有具体到哪一种方法和数据来源）是：无法预测如果用户移动距离很远的情况。
5. 考虑time periodicity, global popularity and personal preference时间周期性，全球流行度和个人偏好的情况下，**实质目标是：预测用户在它们曾经访问过的地点在未来check-in的概率**。
6. 步骤：
   1. 首先通过时间周期性，全球流行度和个人偏好的历史数据来提炼用户的签到模式。
   2. 综合所有因素到一个监督评分模型和一个分类模型中，分别从两个不同的角度解决这个问题。
   3. 并在真实数据上进行验证。达到了0.866的准确率和0.777的F1。
7. 主要贡献
   1. 对12个单独特征的预测能力。
   2. 时间周期和个人偏好在所有特征中的影响力最大。
   3. 将所有特征结合到同一个监督评分模型中，以评估给定用户在给定时间内访问候选地点的可能性。
   4. 通过随机梯度下降算法，设法设法推断出评分模型的参数。
   5. **将预测问题简化为二分类任务。这里有个特别的：训练了两个不同的模型来分类一个给定的用户是否在给定的时间下载一个候选位置check-in。为什么需要训练两个模型**？

## 2. 相关工作

1. 预测远期的签到地点predicting far future check-in location
2. 使用[Hawkes process（霍克斯过程）](https://dreamhomes.top/posts/202106241018/)**来描述check-in动态，这样未来发生的特定事件的可能性就可以通过过去的事件来衡量**。参考文献：Cho, Y.-S., Ver Steeg, G., Galstyan, A., 2014. Where and why users” check in”. In: AAAI,pp. 269–275.

## 3. 问题的定义

1. 研究背景定义了3个层次：社会层、空间层和时间层。
2. 问题1：
   1. 问题1：给定一个精确到小时的时间t，对于所有$L_i$的地点进行排名，以便$u_i$在时间t将访问的确切地点的排名列表中被排在最前面。
   2. 问题2：给定一个时间t和一个特定的地点v，判断$u_i$是否会在时间t和位置v上check-in。
   3. 这两个问题定义了细粒度位置预测问题的不同方面。
3. 11