# translate翻译

1. 题目：Neural Graph Collaborative Filtering
神经图协同过滤

2. 翻译说明：

|number|title of paper|internet source|local source|correlative field|illustration|
|---|---|---|---|---|---|
|1|neural collaborative filtering|<http://staff.ustc.edu.cn/~hexn/papers/sigir19-NGCF.pdf>|/|recommonder system|English translate into chinese|
|2|有人已经翻译好了|<https://www.jianshu.com/p/95da9785bea8>|/|recommonder system|/|

## abstract 摘要

|编号|英语|中文|理解|
|---|---|---|---|
|1|Learning vector representations|学习向量表示法|/|
|2|aka.|又称为|/|
|3|user|用户|这里统一将专有的user翻译为用户。|
|4|item|物品|对应的item翻译为物品。|
|5|attributes|属性|/|
|6|inherent drawback|固有内在缺陷|/|
|7|collaborative signal|协同信号|/|
|8|collaborative filtering effect|协同过滤效果|/|
|9|bipartite graph|二分图|[百度百科说明](https://baike.baidu.com/item/%E4%BA%8C%E5%88%86%E5%9B%BE/9089095)|
|10|propagating embeddings||/|
|11|CCS CONCEPTS|CCS中的概念|The ACM Computing Classification System (CCS)[参考说明](https://blog.csdn.net/qq_40994260/article/details/103741053)。在投稿ACM的时候可以生成。|
|||||

用户和物品的学习向量表示法（又称为嵌入）是当代推荐系统的核心。从早期的矩阵因式分解到最近出现的基于深度学习的方法，现在的主要工作是通过映射预先存在的特征（pre-existing features）来获取user（或者是item）的嵌入向量，将这个嵌入向量来描述为user（或者item），比如ID和属性。我们认为这些方法有固有的缺陷，user和item交互中的潜在的协同信号在嵌入的过程中并没有进行编码（encoded）。因此，生成的embedding可能不足以捕捉协同过滤的效果（collaborative filtering effect）。

在本项工作中，我们将以将user-item交互集成到嵌入过程中（更具体的说是将二分图结构集成到嵌入过程中）。我们开发了一个新的推荐框架：神经图协同过滤（NGCF），它通过propagating embeddings来利用user-item图结构。~~这导致了user-item图中的high-order连通性的表达建模（这句话没有理解）~~，有效的将协同信号以显示的方式注入到嵌入过程中。我们在三个公共基准上进行了广泛的实验，与HOPRec[38]和Collaborative Memory Network[5]等几种最先进的模型相比，取得了显著的改进。进一步的分析验证了嵌入传播对于学习更好的用户和item表示的重要性，证明了NGCF的合理性和有效性。

CCS CONCEPTS
信息系统->推荐系统

关键词
协同过滤，推荐，高阶连通性，Embedding Propagation，图神经网络。
Collaborative Filtering, Recommendation, High-order Connectivity, Embedding Propagation, Graph Neural Network

## 1. 介绍

|编号|英语|中文|理解|
|---|---|---|---|
|1|ubiquitous|无处不在的，形容词|/|
|2|interaction modeling|交互建模，它基于嵌入重建历史交互|/|
|3|deep representations|深度表征|/|
|4|crucial|关键的|/|
|5|argue|动词。认为，主张，发起一个论点的含义。|/|
|6|yield|动词。生成，提供，|/|
|7|sufficient|足够的，充足的，形容词||
|8|satisfactory|令人满意的|/|
|9|reveal|动词，揭示|/|
|10|behavioral|形容词，行为的|/|
|11|descriptive|形容词，描述的|/|
|12|suboptimal|形容词，不理想的，次优的，未达到最佳标准的|/|
|13|deficiency|名词，缺点，不足。|/|

个性化推荐无处不在，已经应用于电子商务、广告和社交媒体等许多在线服务。其核心是根据购买和点击等历史互动来估计用户采纳某个商品的可能性。协同过滤（CF）通过假设行为相似的用户会对item表现出相似的偏好来解决这个问题。为了实现这一假设，一个常见的范例是参数化用户和item以重建历史交互，并根据参数预测用户偏好[1,14]。

一般来说，可学习CF模型有两个关键组件：1）embedding，它将user和item转换为矢量化表示；2）interaction modeling，它基于嵌入重建历史交互。例如，矩阵分解（MF）直接嵌入user/item ID作为向量，并通过内积将user-items建模[20]；通过整合item侧的丰富信息中学习到的深度表征，协同深度学习扩展了MF嵌入功能[29]；神经协同过滤模型用非线性神经网络代替内积的MF交互函数[14]；基于翻译的CF（协同过滤）模型使用(instead)欧几里德距离度量作为交互函数[27]，等等。

尽管这些方法有效，但我们认为，这些方法不足以为CF提供令人满意的嵌入。关键原因是嵌入函数缺乏对关键协同信号的显式编码，而关键协同信号隐藏在user-item交互中，以揭示user（或item）之间的行为相似性。更具体地说，大多数现有方法只使用描述性特征（例如ID和属性）构建嵌入函数，而不考虑user-item交互——这些交互仅用于定义模型训练的目标函数[26,27]。因此，当嵌入不足以捕获CF时，这些方法必须依赖交互函数来弥补次优嵌入的不足[14]。

|编号|英语|中文|理解|
|---|---|---|---|
|1|intuitively|副词，直觉地，直观地。|/|
|2|integrate|动词，整合，合为一体。|/|
|3|trivial|形容词，不重要的。|/|
|4|desired|动词，渴望，期望。|可以翻译为所需的。|
|5|tackle|动词，解决，应对。名词，体育用具。|/|
|6|concept|名词，概念||
|7|denotes|动词，指出||
|8|adopt|动词，采用，接纳||
|9|construct|动词，构造，修建||
|10||||
|||||

虽然直观上可以将user-item交互集成到嵌入函数中，但做好这项工作并非易事。在实际应用中甚至更大，这使得提取所需的协同信号变得困难。在这项工作中，我们通过利用user-item交互的高阶连通性来应对这一挑战，这是一种在交互图结构中编码协同信号的自然方式。

运行示例。[图1说明了高阶连通性的概念](../pictures/NeuralGraphCollaborativeFiltering_Figure1.png "user-item交互图和高阶连通性图。节点$u_1$表示提供推荐的目标用户。")。对于推荐系统$u_1$是用户的兴趣点，在user-item交互图的左侧子图中用双圈标记。右侧子图显示了从$u_1$展开的树结构。高阶连通性指出了一种路径，这种路径可以从任何节点出发只要能够达到$u_1$，而且要求路径长度l大于1。这种高阶连通性包含丰富的语义，承载着协同信号。例如，路径$u_1 \leftarrow i_2 \leftarrow u_2$表示$u_1$和$u_2$之间的行为相似性，因为两个用户都与$i_2$进行过交互；更长的路径$u_1 \leftarrow i_2 \leftarrow u_2 \leftarrow i_4$表明$u_1$可能会接纳$i_4$，因为她相似的用户$u_2$以前也购买过$i_4$。此外，从l=3的整体观点来看，item$i_4$比item$i_5$更有可能引起$u_1$的兴趣，因为有两条路径连接<$i_4$，$u_1$>，而只有一条路径连接<$i_5$，$u_1$>。

现在的工作。我们建议在嵌入函数中对高阶连通性信息进行建模。我们设计了一种神经网络方法，该方法递归地在图上传播embedding信息，而不是将交互图扩展为一棵复杂而且难以执行的树。这是受图神经网络最近发展[8,30,36]的启发，它可以被视为在嵌入空间中构造信息流。具体来说，我们设计了一个嵌入传播层，通过聚合交互项（或用户）的嵌入来细化用户（或item）的嵌入。通过叠加多个嵌入传播层，我们可以强制嵌入以高阶连通性捕获协同信号。以图1为例，将两个层叠加在一起可以捕捉到$u_1$的行为相似性←i2←u2，堆叠三层，捕捉$u_1$的潜在建议← i2← u2← i4和信息流的强度（由层之间的可训练权重估计）决定i4和i5的推荐优先级。我们在三个公共基准上进行了大量实验，以验证我们的神经图协同过滤（NGCF）方法的合理性和有效性。

最后，值得一提的是，尽管最近一种名为HOP Rec[38]的方法考虑了高阶连通性信息，但它仅用于丰富训练数据。具体来说，HOPRec的预测模型仍然是MF，而它是通过优化损失来训练的，该损失由高阶连通性增强。与HOP-Rec不同的是，我们提出了一种新技术，将高阶连通性集成到预测模型中，从经验上看，该技术比HOP-Rec对CF的嵌入效果更好。

## 问题

1. propagating embeddings是在做什么？
2. 构建树模型的复杂度在哪里？
3. 这里所说的embedding propagation layer的作用是否将用户之间访问item的相似性进行了比较？？？
4. 从表示学习的角度来看，pui反映了历史item对用户偏好的贡献程度。 从消息传递的角度来看，考虑到正在传播的消息应随路径长度衰减，因此pui可以解释为折扣因子。
