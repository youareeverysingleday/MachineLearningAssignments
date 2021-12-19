# 学习Generative Adversarial Nets笔记

## 1. 重要知识点

1. GAN是一个无监督学习的算法。
2. GAN的基础结构

   ```mermaid
   graph LR
      start("input Z")-->Generator("G")
      Generator-->output("output")
      output-->Discriminator("D")
      data("data")-->Discriminator
      Discriminator-->Zero("来自于G的输出为0")
      Discriminator-->One("来自于data的输出为1")
   ```

3. 设GAN中的产生器为G，判别器为D。G与D的训练是**交替**进行的。
   1. 第一轮G生成一批output；
   2. D对output和data进行判断；
   3. 得出结果反馈给G（这里没有明白如何反馈给G）；
   4. 在进行第二轮训练。
4. 目标函数$\underset{G}{min}\text{,}\underset{D}{max}\{V(D,G)=E_{x-P_{data}(x)}[logD(x)]+E_{z-p_{z}(z)}[log(1-D(G(z)))]\}$。
5. 原始的GAN无法有效收敛是有原因的。
6. 期望
7. Kullback-Leibler divergence(KL散度)
   1. KL散度用于度量两个分布之间的距离（计算分布之间的相似性）。
   2. G产生的output和data来自同一个分布，但是output不和data中的任何一个样本相同。G最终的目标是学习源数据的分布。
   3. KL散度无法使得GAN收敛。所以换了一种目标评价，这就是Wasserstein散度。
      1. 原因的描述化说明：在高维（低维也一样）空间中随机出现两个流形产生不可忽视的测度不为0的重合的概率为零。这句话的意思举例来说就是：比如在2维平面中随机画2条任意的曲线，它们两者之间产生一段有一定长度的重合的可能性为零。当然可能产生交点，但是对某些不连续的点在计算它们的测度时是为零的。
      2. 所谓测度，举例：给定铁的密度，如果不给定体积，或者说体积为0。那么它的质量测度就是0。
8. Ian之前已经有过一个科学家做了同样的工作。
9. Wasserstein距离。使用Wasserstein距离来代替KL散度来描述两个分布之间的距离。

## 2. 问题

## 3. 其他
