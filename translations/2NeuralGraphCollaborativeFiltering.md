# translate翻译

1. 题目：Neural Graph Collaborative Filtering
神经图协同过滤

2. 翻译说明：

|number|title of paper|internet source|local source|correlative field|illustration|
|---|---|---|---|---|---|
|1|neural collaborative filtering|<http://staff.ustc.edu.cn/~hexn/papers/sigir19-NGCF.pdf>|/|recommonder system|English translate into chinese|

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
|||||

用户和物品的学习向量表示法（又称为嵌入）是当代推荐系统的核心。从早期的矩阵因式分解到最近出现的基于深度学习的方法，现在的主要工作是通过映射预先存在的特征（pre-existing features）来获取user（或者是item）的嵌入向量，将这个嵌入向量来描述为user（或者item），比如ID和属性。我们认为这些方法有固有的缺陷，user和item交互中的潜在的协同信号在嵌入的过程中并没有进行编码（encoded）。因此，生成的embedding可能不足以捕捉协同过滤的效果（collaborative filtering effect）。

在本项工作中，我们将以将user-item交互集成到嵌入过程中（更具体的说是将二分图结构集成到嵌入过程中）。我们开发了一个新的推荐框架：神经图协同过滤（NGCF），它通过propagating embeddings来利用user-item图结构。~~这导致了user-item图中的high-order连通性的表达建模（这句话没有理解）~~，有效的将协同信号以显示的方式注入到嵌入过程中。


## 问题

1. propagating embeddings是在做什么？