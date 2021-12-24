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
|||||
|||||

用户和物品的学习向量表示法（又称为嵌入）是当代推荐系统的核心。从早期的矩阵因式分解到最近出现的基于深度学习的方法，现在的主要工作是通过映射预先存在的特征（pre-existing features）来获取用户的（或者是物品的）的嵌入向量，将这个嵌入向量描述为用户（或者是物品），比如ID和属性。我们认为这些方法有固有的缺陷，用户和物品交互中的隐性协作信号在嵌入的过程中并没有进行编码（encoded）