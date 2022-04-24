# Read the lastest paper quickly

1. 题目：Contextualized Point-of-Interest Recommendation
基于上下文感知和注意的数据增强的POI推荐
2. 相关知识点
   1. [拉普拉斯矩阵和拉普拉斯矩阵正则化](https://blog.csdn.net/weixin_42973678/article/details/107190663)
   2. [邻接矩阵](https://blog.csdn.net/weixin_42265429/article/details/90202076?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165059141216781683942108%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=165059141216781683942108&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-90202076.142^v9^control,157^v4^control&utm_term=%E9%82%BB%E6%8E%A5%E7%9F%A9%E9%98%B5&spm=1018.2226.3001.4187)
   3. [谱聚类](https://zhuanlan.zhihu.com/p/29849122)谱聚类是从图论中演化出来的算法，后来在聚类中得到了广泛的应用。它的主要思想是把所有的数据看做空间中的点，这些点之间可以用边连接起来。距离较远的两个点之间的边权重值较低，而距离较近的两个点之间的边权重值较高，**通过对所有数据点组成的图进行切图，让切图后不同的子图间边权重和尽可能的低，而子图内的边权重和尽可能的高，从而达到聚类的目的**。

   |number|title of paper|internet source|local source|correlative field|illustration|
   |---|---|---|---|---|---|
   |1|Contextualized Point-of-Interest Recommendation|Google学术下载|/|location prediction|distll|

## 问题

1. ...

## 需要进一步了解的

1. ...

## 摘要

|编号|英语|中文|理解|
|---|---|---|---|
|1|employ|动词，使用。应该这个词更书面一点。|/|
|2|property|名词，特质。|/|
|3|alternation optimization method|交替还是交叉优化方法|这个需要通过后文来明确。|
|4|assumption|名词，假设|/|
|5|assume|动词，假设|/|

POI推荐已经成为了推荐系统研究中日益重要的子领域。之前的方法使用多种假设来利用上下文信息以便于提高推荐系统的准确率。它们共同的特点（共同的假设）之一就是：类似的(similar)用户大都访问类似的POI，并且类似的POI都会被相同的(same)用户访问。然而，现有的方法都没有明确的使用相似性来生成推荐。在本文中，我们提出了一个新的POI推荐的框架。具体而言，我们将上下文信息分类为两个groups（暂时还不知道翻译为组好还是翻译为群比较合适），比如全局和本地，将开发不同正则化包含到推荐中。利用图拉普拉斯正则化项来利用全局上下文信息。此外，我们将用户聚类（cluster...into）到不同的groups中，并且让目标函数（objective function）来将有相同预测POI评级的（similar predicted POI ratings）不同用户限制在相同的group中。开发了一个交叉优化方法（alternation optimization method）来优化我们的模型，并且获得一个最终的评级矩阵。我们的实验结果显示我们的算法优于其他所有的SOTA方法。

## 1. 介绍

|编号|英语|中文|理解|
|---|---|---|---|
|1|aim to|目标在于...|/|
|2|interesting spot|兴趣地点|这里点的点没有使用point，而是使用了spot。spot更倾向于表示地点。|
|3|unfamiliar|形容词，不熟悉的。|/|
|4|trackle|动词，应付，解决。|/|
|5|construct|动词，创建、建立、组成。|/|
|6|auxiliary|形容词，辅助的|/|
|7|accessory component|配件|/|

POI推荐已经成为了推荐系统研究中日益重要的子领域，并且POI推荐的目标在于为用户找到可能感兴趣的新位置。它能帮助用户找到兴趣点从而帮助他们在不熟悉的地域享受他们假期。同时POI推荐也可以通过吸引更多愿意花时间和金钱在商店的客户来增加店主们的收入。因此，在最近几年POI推荐变成了一个热门研究主题。然后，POI推荐中依然面临很多挑战，其中最重要的挑战就是数据稀疏性。

为了解决这个问题，许多方法都通过不同的假设将上下文信息包含到推荐方法中。举例：IRenMF[Liu et al.,2014b]的假设用户将访问的新POI是他们之前访问的POI附近。并且他们通过添加对每个POI邻近POI的评分加权和建立了辅助的标签矩阵（auxiliary label matrix）。LTR[Gao et al., 2013]假设在不同的时间隙（time slot）中将会有不同的偏好模型，因此他们对不同的时间周期建立了不同的模型。虽然假设是多种多样的，共同的特征：类似的(similar)用户大都访问类似的POI，并且类似的POI都会被相同的(same)用户访问。这些假设之间唯一的不同在于构造相似性的方法，举例而言，IRenMF中使用的相似性时地理距离，LRT中使用的相似性是时间差。

虽然，它们在使用上下文信息的方式上有两个主要的缺点。其一它们通常只在一个实体中考虑一种类型的上下文信息，比如POI之间的地理距离或者用户之间的友谊。并且它们专门为特定的类型的场景设计模型，导致模型的可扩展性比较弱。另一个问题是它们没有准确的使用上下文信息，大多数模型关注于check-in的历史，导致上下文信息的使用只是作为目标函数的附属品（accessory component）。这样，就导致上下文信息作为POI推荐性能的关键点无法得到充分的使用。

注：读到这里主要的改进是是为用户和POI创建了矩阵，而之前的用的是向量，也就是用向量表示了用户和POI之后就通过模型了。而这一片是将user和POI变成了矩阵。

|编号|英语|中文|理解|
|---|---|---|---|
|1|constrain|动词，约束、限制、强迫|/|
|2|hierarchically|副词，分层、谱系|/|
|3|impose|动词，采用|/|
|4|local|形容词，可以范围以本地的也可以翻译为**局部的**，这里就翻译为局部的为宜|/|
|5|pattern|名词，模式|/|
|6|spectral clustering|谱聚类|/|

为了模型能够可扩展，并且能够充分利用上下文信息，我们提出针对POI推荐提出了一个新的框架。在我们的方法中，我们按照相应的user和POI上下文信息构建了一个用户矩阵和一个POI相似矩阵。很多类型的相似性可以通过两个实体之间的特征向量的余弦相似性来计算。此外，不同类型的相似性可以通过组合为加权和。这样，我们框架对于一大类上下文信息（a large class of contextual information）而言是可扩展。一旦用户和POI的相似矩阵能被创建，**我们将使用两个全局拉普拉斯正则化项来约束预测偏好矩阵**。那样能够直接确定，最终的预测：相似的用户应该访问相似的POI，并且相似的POI会被相似的用户访问。另外，分等级的使用上下文信息，我们也采用了一个局部正则化来使得预测偏好矩阵拥有局部模式。基于用户相似性，我们使用谱聚类[Von Luxburg, 2007]将用户排序放入不同的group中。然后我们对每个组的预测偏好矩阵施加一个$l_2 -norm$作为（as 翻译为作为）正则化项，这样可以使得在同一个组中的用户的偏好稀疏（to be sparse，这样翻译有点奇怪）且具有相同的模式。

|编号|英语|中文|理解|
|---|---|---|---|
|1|Accelerated proximal gradient (APG) algorithm|加速近端梯度算法|/|
|2|exploit|动词，剥削、开发、利用|/|
|3|explicit|形容词，明确的、清晰的|/|

我们通过将全局和局部正则化放在一起来构造目标函数。为了有效的解决优化问题的，我们提出了一个交替优化方法，该方法将目标函数分解为两个部分，并带有一个辅助变量。Accelerated proximal gradient (APG) algorithm被用于优化这个问题的$l_2 -regularized$部分。

本项工作的贡献如下：（1）我们提出了一个对于POI推荐的新框架，这个框架聚焦于上下文信息的清晰使用；（2）在我们的方法中，我们能同时使用user和POI的不同类型的上下文信息；（3）我们将上下文信息按照全局类型和局部类型进行分类，并通过各自的不同的正则化项来使用它们；（4）我们设计了一个交替优化方法来优化模型；（5）在两个大规模的数据集上我们方法的结果胜过其他SOTA方法。

## 2 相关工作

多种方法已经提出如何在POI推荐中使用上下文信息。一组方法使用基于用户的上下文。在RankGeoFM[Li et al.,2015]中，作者为每个用户提出了两个不同的隐性特征，一个是对于目标POI的偏好，另一个是对于邻近POI的偏好。另外一组方法聚焦于基于POI的上下文。例如，IRenMF[Liuet al., 2014b]利用不同POI之间基于地理距离[Liu et al., 2014b]的相似性。对于每个POI和特定用户，它使用邻近POI的用户偏好的权重和来评估POI的评分。并且它利用相似的模式来约束同一地域内的POI的隐性特征。尽管它仅仅使用POI之间的地理距离，按照最新的综述[Liu et al., 2017]中说明这个工作的性能是最好的。此外，依然有些方法对于推荐使用了混合上下文。例如，GeoMF[Lian et al., 2014]将整个空间分割为不同地域，并且它将含有用户的兴趣和这些地域的POI影响放入他们的模型中。然后最终的偏好评分由独立的和区域性成分组成。

## 3 准备工作

本节介绍一些关于POI推荐问题的和图拉普拉斯正则化的背景知识。

### 3.1 问题定义

|编号|英语|中文|理解|
|---|---|---|---|
|1|set of indices|索引集|/|
|2|transaction|名词，处理、交易|/|
|3|geographical coordinates|地理的坐标|/|
|4|setting|名词，环境|/|
|5|serves as|作为|/|
|6|convenience|名词，方便|/|
|7|represented|动词，代表;作为…的代言人;维护…的利益;等于;相当于|/|

假设在推荐任务中用户的总数为m，POI的总数是n。设$U=\{u_1, u_2, \cdots, u_m\}$为用户集，设$V=\{v_1, v_2, \cdots, v_n\}$，设$P_u$是用户$u(u \in U)$访问的POI的索引集。给定过去的check-in业务历史$D$，POI推荐的任务是将POI推荐给每个用户u，一个新的POI索引集$\hat{P}_u, \quad (P_u \bigcap \hat{P}_u = \emptyset)$来匹配他们的（用户的）偏好。check-in业务历史$D$是用户和他们访问过的POI的元组，比如$D=\{(u, v)| u \in U,\, v \in V\}$。用户（比如社交关系）和POI（比如地理坐标）的上下文信息可以被推荐系统使用来生成推荐。在典型的有监督学习环境中，$D$作为训练集。一组额外的业务集$D^{Te}$，包含推荐系统中生成的不被每个用户可见的check-in数据，它服务于测试集。设$P_u^{Te}$表示为在测试集中由用户u访问过的POI的索引集。推荐的质量可以通过$\hat{P}_u$和$P_u^{Te}$之间交叉的大小来衡量。为了方便，业务历史集$D$也是一个评分矩阵（rating matrix）$Y,\; Y_{ij}=1 \quad if(u_i, v_j) \in D \quad \text{and} (Y)_{ij}=0 \quad if(u_i, v_j)\notin D$。

### 3.2 图拉普拉斯正则

|编号|英语|中文|理解|
|---|---|---|---|
|1|weighted undirected graph|加权无向图|/|
|2|vertex|名词，顶点|/|
|3|symmetric matrix|对称矩阵|/|
|4|diagonal matrix|对角矩阵|/|
|5|degree matrix|次数矩阵|/|
|6|$diag(d_1, d_2, \cdots, d_{|V|})$|/|表示一个对角矩阵，对角线上的元素是$(d_1, d_2, \cdots, d_{|V|})$|
|7|real-valued function|实值函数|/|
|8|quadratic form|二次型|/|
|9|applying function|应用函数|/|
|10|formed by|由...形成的|/|

设$G$是一个加权无向图，其中$V=\{v_1, v_2, \cdots, v_{|V|}\}$是它的顶点集，并且它的权重矩阵是$\boldsymbol{W}=[W_{ij}]_{i,j=1,2,\cdots,|V|}$，其中$W_{ij}$表示在$v_i$和$v_j$之间边的权重。因为$G$是一个无向的，所以$\boldsymbol{W}$是一个对称矩阵，比如$W_{ij} = W_{ji}$。G的次数矩阵（degree matrix）$\boldsymbol{D}$是一个对称矩阵，$diag(d_1, d_2, \cdots, d_{|V|}),\; \text{其中}d_i=\sum\limits_{j=1}^{|V|}W_{ij}$。然后，G的规范化拉普拉斯矩阵被定义为$\boldsymbol{L}=\boldsymbol{I} - \boldsymbol{D}^{-\frac{1}{2}}\boldsymbol{W}\boldsymbol{D}^{-\frac{1}{2}}$。设$f:V\rightarrow \mathbb{R}$是一个定义在顶点空间$V$上的实值函数。在图$G$中$f$规范化拉普拉斯正则化定义为二次型：
$$\boldsymbol{L}_f= f(V)^T \boldsymbol{L}f(V) \tag{1}$$
其中，$f(V)=[f(v_1),f(v_2), \cdots, f(v_{|V|})]^{\top}$是通过在顶点空间$V$上的应用函数$f$而形成的向量。

## 4 建议的推荐模型

|编号|英语|中文|理解|
|---|---|---|---|
|1|represent|动词，表示，代表|/|
|2|infer|动词，推断、推理|/|
|3|carry out|执行、完成、展开|/|

本节说明了建议的推荐模型框架的细节。我们模型的目标是通过优化一个目标函数来预测一个评分矩阵$\boldsymbol{R} \in \mathbb{R}^{m \times n}$。在$\boldsymbol{R}$中的每个元素$R_{i,j}$表示用户$u_i$对于POI $v_j$的推断偏好。对于用户$u_i$而言会基于$R_{i,1},R_{i,2}, \cdots, R_{i,n}$的值推荐一些新的POI。目标函数包含3个在$\boldsymbol{R}$中的正则项，并且每个正则项都将会各自（respectively）显式地利用基于用户的全局上下文信息、基于POI的全局上下文信息和局部上限为信息。最终，我们说明了不同的正则化项是如何组合在一起的，以及如何进行优化的。

### 4.1 使用全局上下文信息

|编号|英语|中文|理解|
|---|---|---|---|
|1|incorporate|动词，合并、包含、将…包括在内、吸收。|这里使用包含比较合适|
|2|particular|形容词，专指的，特指的(与泛指相对);不寻常的;格外的;特别的;讲究;挑剔|/|

基于用户的上下文：在我们的框架中包含了基于用户的全局上下文信息，我们假设对一个特定POI类似的用户有类似的评分。通过基于用户的全局上下文信息来计算用户的相似性（比如：用户之间的社交关系信息）。设$G_{user}$是一个带权值的无向图，其中顶点集是用户集$U$。通过一个对称权值矩阵（symmetric weight matrix）来给定边的权重$\boldsymbol{W}_{user}=[W_{ij}^{user}]_{i,j=1,2,\cdots ,m}$，其中$W_{ij}^{user}$表示了用户i和用户j之间的相似性。这里我们假设$W_{ij}^{user}$是给定的，并且如何通过基于用户的全局上下文信息的特定类型来构建$W_{ij}^{user}$的详细说明将在第4.3节中说明。正如3.2节中所述，$G_{user}$的次数矩阵是$\boldsymbol{D}_user=diag(d_1^{user},d_2^{user}, \cdots , d_m^{user})$并且$G_{user}$的归一化拉普拉斯矩阵是$\boldsymbol{L}_{user}=\boldsymbol{I}-\boldsymbol{D}_{user}^{-\frac{1}{2}}\boldsymbol{W}_{user}\boldsymbol{D}_{user}^{-\frac{1}{2}}$。假设评分矩阵$\boldsymbol{R} \in \mathbb{R}^{m \times n}$，其中$R_{ij}$表示的是用户$u_i$在POI$v_j$上的评分。对于一个特定的POI$v_j$，我们在$G_{user}$上$\boldsymbol{R}$的归一化图拉普拉斯正则化定义为$\mathcal{L}_{user}(\boldsymbol{R}_{:,j})=\boldsymbol{R}_{:,j}^{\top}\boldsymbol{L}_{user}\boldsymbol{R}_{:,j}$，其中“:”表示获取行/列的所有项，然后我们有
$$\mathcal{L}_{user}(\boldsymbol{R}_{:,j})=\sum\limits_{i=1}^{m}\sum\limits_{k=1}^{m}W_{ik}^{user}[\frac{R_{ij}}{\sqrt{d_{i}^{user}}}-\frac{R_{ik}}{\sqrt{d_{k}^{user}}}]^2 , \tag{2}$$

从等式(2)中可以清晰的看到我们将$\mathcal{L}_{user}(\boldsymbol{R}_{:,j})$纳入损失函数中，它将促进更多的相似用户在POI$v_j$上有相似的评分。从这个中观察到，我们对于在图$G_{user}$上的所有的POI的$\boldsymbol{R}$的归一化图拉普拉斯正则化项求和，得到
$$\mathcal{L}_{user}(\boldsymbol{R})=\sum\limits_{j=1}^{n}\mathcal{L}_{user}(\boldsymbol{R}_{:,j}) = trace(\boldsymbol{R}^{\top}\boldsymbol{L}_{user}\boldsymbol{R}), \tag{3}$$

它可以用作评分矩阵$\boldsymbol{R}$的基于用户的拉普拉斯正则化项。等式(3)明确的使用上下文信息来促进在所有的POI上相似的用户有相似的评分。

基于POI的全局上下文：基于POI全局上下文信息的使用方式与它对应的基于用户上下文信息类似。对于基于POI全局上下文信息，我们可以像为用户所做的那样，在POI空间中构建类似的图。我们能通过相同的方式来定义$G_{poi}, \boldsymbol{W}_{poi}, \boldsymbol{D}_{poi} \text{和}\boldsymbol{L}_{poi}$。对于一个给定的评分矩阵$\boldsymbol{R}$，在$G_{poi}$上对于一个特定用户$u_i$它的归一化拉普拉斯正则化项是$\mathcal{L}_{poi}(\boldsymbol{R}_{i,:}) = \boldsymbol{R}_{i,:}\boldsymbol{L}_{poi}\boldsymbol{R}_{i,:}^{\top}$。与等式(3)相同，对于所有用户而言基于POI的拉普拉斯正则项为
$$\mathcal{L}_{poi}(\boldsymbol{R}) = \sum\limits_{i=1}^{m}\mathcal{L}_{poi}(\boldsymbol{R}_{i,:})=trace(\boldsymbol{R}\boldsymbol{L}_{poi}\boldsymbol{R}^{\top}), \tag{4}$$

作为评分矩阵R的基于POI的拉普拉斯正则化项。这个拉普拉斯正则化明确促进所有用户对类似的POI进行类似的评级。

### 4.2 使用局部上下文信息

|编号|英语|中文|理解|
|---|---|---|---|
|1|explicitly|副词，明白地，明确地|/|
|2|exploit|动词，开发，利用，剥削，发挥|/|
|3|enumerated|动词，列举、枚举|/|
|4|take into account|考虑到|/|
|5|assumption|名词，假设、假定|/|
|6|denote ... as ...|将...表示为...|/|
|7|lasso|Least absolute shrinkage and selection operator最小绝对收缩和选择算子|/|
|8|argument|名词，论点;争论;论据;辩论;争吵;争辩;理由|/|
|9|||/|
|10|||/|

在4.1节的介绍中正则项能有效的利用用户和POI之间的全局上下文信息。我们能通过等式(2)来理解“全局”的含义，其中**对**用户都被列举出来，它们的平方差会求和，并且通过任何用户对之间存在的全局性相似性评分$\boldsymbol{W}_{ik}^{user}$进行调整。在这一节中，我们介绍另一个类型的正则化器，这个正则化器考虑到用户和POI之间的局部上下文信息。这个局部正则化器也是基于相似用户对相似POI有相似的评分。但局部正则化器将用户分离到组中，因此导致相似性时“局部的”。为了将相似的用户分配到组中，我们在用户相似性图($G_{user}$)的拉普拉斯矩阵($\boldsymbol{L}_{user}$)上使用了谱聚类（spectral clustering）[Von Luxburg, 2007]。假设cluster的总数是$G$。我们通过在第g个cluster的用户的POI$v_j$的评分表示为$\boldsymbol{R}_{(g),j}$。局部正则化器$\mathcal{J}(\boldsymbol{R})$定义如下：

$$\mathcal{J}(\boldsymbol{R}) = \sum\limits_{g=1}^{G}\sum\limits_{j=1}^{n}\omega_g||\boldsymbol{R}_{(g),j}||_2 , \tag{5}$$

其中$\omega_g = \sqrt{n_g}$，并且$n_g$是在cluster$g$中用户的数量。这个正则化器是一个$\it{group\;lasso\;regularizer}$，在[Yuan and Lin, 2006]中说明$\it{group\;lasso\;regularizer}$，并且非常广泛的应用[Jenatton et al., 2010; Kim and Xing, 2010;Kolar et al., 2009]。相同的观点

### 4.3 相似图构造

物品相似图构造

用户相似图构造