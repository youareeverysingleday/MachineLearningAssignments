# 相关知识点笔记

Embedding

## 参考

1. <https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tutorials/text/word_embeddings.ipynb>原理部分辅助理解，实现部分没有说清楚。
2. <https://tensorflow.google.cn/text/guide/word_embeddings>是1的英文原版。
3. <https://zhuanlan.zhihu.com/p/46016518>这篇只做了介绍，并没有说得很深入。
4. <https://www.zhihu.com/question/38002635>原理说明的非常清楚。
5. <https://zhuanlan.zhihu.com/p/138310401>这篇好像是所有原理的来源。其中把3种实现方法也说清楚了。
6. <https://zhuanlan.zhihu.com/p/85802954>主要是有些使用tensorflow代码的部分。
7. <https://www.bilibili.com/video/BV1Cf4y1e7Ht?from=search&seid=16604717378042548315&spm_id_from=333.337.0.0>来**详细说明embedding的过程，这一片视频对理解非常有用**。

## 理解

1. **容易理解**有个举例的例子对于理解embedding非常有用：[就是通过RGB3色来表示所有的颜色](https://www.zhihu.com/question/38002635)。**一句话解释：用低维的向量表示高维空间的图形**。我们已经知道表示颜色的三个维度有明确对应的物理意义（即RGB），直接使用物理原理就可以知道某一个颜色对应的RGB是多少。但是对于词，**我们无法给出每个维度所具备的可解释的意义，也无法直接求出一个词的词向量的值应该是多少**。所以我们需要使用语料和模型来训练词向量——把嵌入矩阵当成模型参数的一部分，通过词与词间的共现或上下文关系来优化模型参数，最后得到的矩阵就是词表中所有词的词向量。这里需要说明的是，有的初学者可能没绕过一个弯，就是“最初的词向量是怎么来的”——其实你只要知道最初的词向量是瞎JB填的就行了。嵌入矩阵最初的参数跟模型参数一样是随机初始化的，然后前向传播计算损失函数，反向传播求嵌入矩阵里各个参数的导数，再梯度下降更新，这个跟一般的模型训练都是一样的。等训练得差不多的时候，嵌入矩阵就是比较准确的词向量矩阵了。
2. 数学解释：**一句话解释：就是将高维空间的图形通过低维空间的形状进行归纳和解释。这其中就需要流形的概念来进行说明**。Embedding（嵌入）是拓扑学里面的词，在深度学习领域经常和Manifold（流形）搭配使用。可以用几个例子来说明，比如三维空间的球体是一个二维流形嵌入在三维空间（2D manifold embedded in 3D space）。之所以说他是一个二维流形，是因为球上的任意一个点只需要用一个二维的经纬度来表达就可以了。又比如一个二维空间的旋转矩阵是2x2的矩阵，其实只需要一个角度就能表达了，这就是一个一维流形嵌入在2x2的矩阵空间。什么是深度学习里的Embedding？这个概念在深度学习领域最原初的切入点是所谓的Manifold Hypothesis（流形假设）。流形假设是指“自然的原始数据是低维的流形嵌入于(embedded in)原始数据所在的高维空间”。那么，深度学习的任务就是把高维原始数据（图像，句子）映射到低维流形，使得高维的原始数据被映射到低维流形之后变得可分，而这个映射就叫嵌入（Embedding）。比如Word Embedding，就是把单词组成的句子映射到一个表征向量。但后来不知咋回事，开始把低维流形的表征向量叫做Embedding，其实是一种误用。如果按照现在深度学习界通用的理解（其实是偏离了原意的），Embedding就是从原始数据提取出来的Feature，也就是那个通过神经网络映射之后的低维向量。
3. 实现的过程：**和训练神经网络一样，也是训练一组权重值**，开始的时候随机初始化，然后在训练过程中通过反向传播来调整这一组权重值。嵌入向量的权重会随机初始化（就像其他任何层一样）。在训练过程中，通过反向传播来逐渐调整这些权重。训练后，学习到的单词嵌入向量将粗略地编码单词之间的相似性（因为它们是针对训练模型的特定问题而学习的）。其实你只要知道最初的词向量是瞎JB填的就行了。嵌入矩阵最初的参数跟模型参数一样是随机初始化的，然后前向传播计算损失函数，反向传播求嵌入矩阵里各个参数的导数，再梯度下降更新，这个跟一般的模型训练都是一样的。等训练得差不多的时候，嵌入矩阵就是比较准确的词向量矩阵了。
4. 举例理解：[1,2,3,4,5]这是一个原始的共有5个类型的输入数据。由于这个向量是有连续的值整数值来表示的，它们在逻辑上是没有联系的。此时可以通过embedding将它们尝试通过一个3维的向量来表示这5个整数值代表的分类。具体形式就是[[0.7, 0.2, 0.1], [0.4, 0.4, 0.2], [0.3, 0.5, 0.2], [0.1, 0.4, 0.5], [0.1, 0.2, 0.7]]。具体的值只是举例，不是实际计算的值。这样就将一个5种类型的事物通过3种类型进行了表示，而且在表示的过程中包含了它们之间的关系。详细代码见下面的例子及说明。下面的例子是[tensorflow的官方示例](https://tensorflow.google.cn/api_docs/python/tf/keras/layers/Embedding?hl=en)，为了方便理解对当中的一些值做了修改，并添加了注释。可以运行成功。

    ```python
    import tensorflow as tf
    import numpy as np
    # 需要注意的是tf.keras.layers.Embedding要求的输入是一个2维矩阵，输出的是一个3维矩阵。好像这里感觉没有进行数据压缩，实际上需要深入的理解tensorflow的Embedding是在哪个位置进行的降维。2维矩阵中每个元素的取值范围表示了总的类型，而输出的3维矩阵的第3个维度的大小表示了降维的大小。
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=32, output_dim=3, input_length=2))
    # 这个Embedding3个主要参数说明：
    # input_dim用于表示输入数据的维度。如果是分类问题，那么这个值就是样本总的类别数量。
    # output_dim表示的输出数据的表示维度。如果是分类问题，那么这个值就是在输出的时候希望用几个维度的数值来表示所有类别中一个种类(一个样本)。
    # input_length表示的是输入数据的表示维度。如果是分类问题，那么就表示用几个维度的数据来表示单个种类。这个参数是可选参数，也就是说tf.keras.layers.Embedding会自动识别input_length的值，或者说自动识别表示单个种类的维度数。
    # The model will take as input an integer matrix of size (batch,
    # input_length), and the largest integer (i.e. word index) in the input
    # should be no larger than 999 (vocabulary size).
    # Now model.output_shape is (None, 10, 64), where `None` is the batch
    # dimension.
    input_array = np.random.randint(32, size=(16, 2))
    # 这是生成数据的位置。其中32可以理解为总的样本空间中样本的总的类型是32种。size=(16, 2)这里可以为每个样本是通过一个长度为2的向量来表示的；总共有16个这样的样本，也就是说将会有16个样本用于训练。
    # np.random.randint需要和Embedding中的参数对应上。32就是input_dim，2就是input_length。16是样本数量，所以在Embedding的参数中没有体现。
    print(input_array)
    print(input_array.shape)

    model.compile('rmsprop', 'mse')
    output_array = model.predict(input_array)

    print(output_array.shape)
    print(output_array)
    ```

## 重要知识点

1. 来源：需要可靠的表示离散变量。比如文字中的单词或者单字。
2. 其他的表示方法的优缺点：
   1. 独热编码：过于稀疏。
   2. 唯一的数字对每个单词编码：
      1. 解决了稀疏的问题。
      2. 不能表示单词之间的关系；
      3. 对模型进行解释的时候学习到的权重没有含义。
3. 在2的基础上需要一种**高效、密集、能表示关系、能自动编码**的表示方法。
4. 生成Embedding的方法可以归类为3种，分别是矩阵分解，无监督建模和有监督建模。
5. 在无监督建模中最经典的方法是word2vec。
6. 相比RGB，Embedding最大的劣势是无法解释每个维度的含义，这也是复杂机器学习模型的通病。
7. embedding层其实是一个全连接神经网络层。最后的输出结果就是一个代表离散量的向量。
8. embedding在数学上是建立一种X到Y的单向映射。在embedding的过程中一般是有损失的。
9. 狭义的embedding是将离散的值投影到一个连续的向量空间。
10. 在embedding的过程中期望是保持数据的结构。
11. embedding使用的假设是：任何事物都有一些简单的内在联系。或者说任何事物都可以通过一些简单的关系来表示其发展。没有办法去证明这种情况存在，但是这就是一种非常好用的假设。
12. **应用场景：将一个离散的、稀疏的、高维度的分类数据变到一个连续的、稠密的、低维度的数据**。
    1. 文本单词的处理。
    2. 图像也可以embedding。embedding之后对图像的相似度进行比较。
    3. 数据的可视化。人对超过3维以上的数据是无法理解的，可以将高维的数据降低到低维度上方便人的理解。representation learning表征学习。
13. 推荐系统中的基本假设：每个用户之间都是有相似性的，而不是每个用户都是独立存在的。同时，每个商品之间也是存在某种相似性的。
14. user bias实际上就是用户独有的一些特征。
15. 存在的问题：
    1. 不能保证训练出来的问题是可以应用到其他的问题中。
    2. 增加的模型的参数。
    3. **embedding实际上做了一个线性的变换**。对事物的投影是有局限性的，不一定能够实现对比较复杂的关系的认知。
16. 一个比较典型的词向量嵌入模型word2vec，实际上就是将词嵌入到一个向量空间中。word2vec实际上也是线性的。只不过采用的方法不同。
17. 严格的来说word2vec是一种监督学习（无论是对CBOW或者skip gram两种算法而言），都对词做了各种格言的标注。在实际应用中并没有对导入的数据进行标注，而是模型自己对数据进行了处理。所以，一般就把这种模型定义为了自监督学习。
18. 严格意义上的无监督学习：auto encode。这个和镜像预训练是一样的。就是比较输入和输出的相似性，auto encode是得到中间的隐藏层为embedding，镜像预训练是得到网络中的权重作为模型训练之前的初始值。auto encode应用的场景包括：训练表达、图像降噪。
    1. 在图像降噪应用中，给定一个高清图，然后人工给高清图添加噪声，然后需要将输入和输出的图片之间的差距越小越好。
    2. sparse auto encoder，它强调的是auto encoder得到的结果必须是一个稀疏的网络。也就是每一个神经元不是零的概率非常小。衡量一个样本空间的稀疏性通过sigmoid激活函数，把输入控制在0和1之间，然后在把输出相加起来，再除以样本个数，得到的值可以近似的理解成网络的稀疏性。通过估计的稀疏性和想要的稀疏性之间的差别来作为损失函数的一部分去做训练。通常来说通过KL divergence来衡量两个分布之间的不同情况（它的斜率更陡），从而更好的关注网络的稀疏性。
    3. auto encoder不光是可以去学习表示事物本身的表示，它还可以去学习事物的分布情况。在这个过程中可能会产生一些随机的东西，有可能和原来的事物很像，但又不是完全一样。产生新事物的generative learning和representation learning（表示学习）。两者之间有很多内在的联系。

-- END --

## 问题

1. auto encoder和PCA是类似的东西，如何理解的？
    - auto encoder中有个部分是也是在做降维的操作，他们的实现机理有些类似。
