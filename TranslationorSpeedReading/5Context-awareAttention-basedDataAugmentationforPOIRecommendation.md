# 速读

1. 题目：Context-aware Attention-based Data Augmentation for POI Recommendation
基于上下文感知和注意的数据增强的POI推荐

|number|title of paper|internet source|local source|correlative field|illustration|
|---|---|---|---|---|---|
|1|Learning Graph-based POI Embedding for Location-based Recommendation|<https://arxiv.org/pdf/2106.15984.pdf>|/|location prediction|distll|

## 需要做的事情

## 总结

这篇文章做的是补充一个序列中可能check-in但是没有check-in的位置。有点类似于特征工程中的填缺失值。使用Seq2Seq方法对稀疏的数据进行填充。

## 摘要

本文提出来了一个基于注意力的序列到序列的生成模型，即POI-Augmentation Seq2Seq (PA-Seq2Seq)，通过使check-in记录的间隔均匀来解决训练数据集的稀疏性问题。具体而言，编码器对每个check-in序列进行总结，解码器根据编码信息预测可能丢失的check-in。为了学习user check-in历史中与时间上下文的相关性，我们采用了局部注意力机制来帮助解码器在预测某个丢失的check-in point时关注特定范围的上下文信息。我们在Gowalla和Brightkite这两个真实check-in数据集上进行了广泛的实验，以评估信息和效果。

关键词：数据增强、兴趣点、POI推荐 Data Augmentation, P Point-of-interest, POI Recommendation

## 简介

本文完成的是按照综述中的分类和简介中的说明属于：next POI recommendation  下一个POI的推荐。也就是说不是实时的。

通过check-in中包含的文本、时间和地理信息来使得基于位置的服务提供商为用户提供可定制的更准确地客户营销策略。

最近关于序列数据的分析和推荐、隐性表示模型和马尔科夫链的工作或研究已经被广泛探讨。[1]是因式分解个性化马尔科夫链FPMC（Factorizing Personalized Markov Chain）。[2]添加了了对FPMC的用户移动约束和设定next POT推荐的数量来发展FPMC。

上述方法都受到了数据稀疏性问题的负面影响。相比于对电源和购物的评分，用户更在每个到达的POI位置上check-in的概率更低。因此，在连续签到之间可能丢失许多可能的check-in信息。

据我们所知，这是第一个关于next POT推荐任务的数据增强的研究。如图1所示，我们的任务时将可能缺失的check-in数据插入训练集，使其在时间间隔上均匀分布。最后，帮助next POT推荐模型更好的理解用户的偏好和行为模型。传统上，数据扩充的一个常见方法是线性插值，在本任务中，计算点p在两个观察到的POI之间的直线上的地理位置，并选择p最近的POI  l作为预测的缺失check-in。然后这种方法是不可靠的，因为由于地理上的胭脂和时间上的偏好，用户的轨迹路径更可能是一条曲线而不是直线。

--END--