# TorusE: Knowledge Graph Embedding on a Lie Group    speed read

## 摘要

|编号|英语|词性|本文中翻译的中文|理解|
|---|---|---|---|---|
||principle|n|道德原则;行为准则;规范;法则;原则;原理|/|
||sphere|n|球体；球形|/|
||fulfill|v|履行，执行，贯彻；实现，完成|/|
||adversely|adv|不利地;反而;反对地|/|
||diverge|vi|发散|偏离;分歧;岔开;分叉;相异;背离;违背|
||novel|adj|新颖的;与众不同的;珍奇的|做名词时是长篇小说的意思。|
||torus|n|圆环体|/|
||scalable|adj|可扩展的|/|
||inference|n|推理|/|
||hence|adv|因此|/|
||triple|n|三元组|这里是知识图谱中的特指。一般翻译为三倍。|
||outerform|v|胜过|/|

知识图谱对于很多人工智能任务来说都是有用的。尽管，知识图谱经常缺失事实。为了填充图，开发了知识图谱嵌入模型。在一个知识图谱中，知识图谱嵌入模型将实体和关系映射为到一个向量空间，并且通过对候选三元组的评分来预测位置的三元组。在我们所知的范围内，对于知识图谱嵌入的完成，TransE是第一个基于翻译（translation-based model）的模型，并且它是简单、有效。它采用了实体嵌入之间的差别来表示它们关系的原理。这个原理看起来非常简单，但是它能够有效的捕捉一个知识图谱的规则。尽管TransE存在正则化的问题。TransE强制将实体嵌入到一个嵌入向量空间的球体上。这种正则扭曲了嵌入，使得它们难以实现上面提到的原理。这种正在对link prediction的精确度也有不利影响。在其他方面，正则很重要，因为实体嵌入在没有正则化的情况下会因为负采样而发散（diverge）。本文提出了一个新颖的嵌入模型，TorusE，来解决正在问题。TransE的原理能被定义在任何李群上。为了避免正则化，嵌入空间可以选择一个紧李群（compact Lie groups）环面（a torus）。在我们所知范围内，TorusE是第一个将对象嵌入到真实或者复杂向量空间之外的模型，并且本论文第一次正式的讨论了TransE的正则化问题。我们的方法在标准的link prediction task中胜过了包括TransE、DistMult、ComplEx、在内的所有达到SOTA的方法。我们展示了TorusE在一个大尺寸的知识图谱上的可扩展性，并且展示了TorusE比最初的TransE更快。

## 1. 简介

知识图谱是一种在一个计算机能够轻松处理情况下描述真实世界的方法之一。比如YAGO、DBpedia和Freebase等知识图谱已经被用于很多任务中，比如问答、内容标记、事实检查和知识推理。尽管一些知识图谱包含百万级别的实体和十亿几倍的事实，他们依然可能不完整并且确实很多事实。因此，知识图谱需要开发一个能够自动补全知识图谱的系统。

|编号|英语|词性|本文中翻译的中文|理解|
|---|---|---|---|---|
||strongly|adv|强烈地，坚决地 ; 大力地 ; 浓烈地|这里翻译为密切地非常合适|
||conflict|n/v|冲突;争执;争论;(军事)冲突;战斗;抵触;矛盾;不一致/(两种思想、信仰、说法等)冲突，抵触|conflict with与...冲突|

在一个知识图谱中，事实被存储在一个有向图中。每个顶点（node）表示在真实世界中的一个实体，每一条边标识实体之间的关系。一个事实通过一个三元组（triple）$(h,r,t)$来表示，其中$h,t$是实体，$r$表示从$h$指向$t$的关系。有一些关系是密切的关系。举例而言，关系$HasNationality$和关系$CityOfBirth$是有关的。因此，如果在知识图谱中三元组（$DonaldJohnTramp, HasNationality, U.S.$）没有存储，而（$DonaldJohnTramp, CityOfBirth, NewYorkCity$）三元组存在其中，知识图谱创建之能够非常容易的预测出$HasNationality$关系，这是因为绝大多数在纽约出生的人拥有美国国籍。很多模型已经能够预测位置的三元组，并且能够通过link prediction任务预测缺失的$h$或者$t$来补充知识图谱。

对于link prediction任务而言，TransE是最早基于翻译的模型，由Bordes等人在2013年提出，因为它简单而且有效，因此它非常有名。如果已经通过训练数据将三元组$(h,r,t)$存储在了知识图谱中；那么TransE通过$\boldsymbol{h}+\boldsymbol{r}=\boldsymbol{t}$的原理来将三元组和关系嵌入在一个真实的向量空间上，其中$\boldsymbol{h}$、$\boldsymbol{r}$和$\boldsymbol{t}$分别是$h, r, t$的嵌入表征。虽然它非常的简答，但是这个这个原理能够非常有效的捕捉知识图谱中的结构。已经提出很多TransE的扩展版本。它们包括TransH(Wang et al. 2014)、TransG(Xiao, Huang, and Zhu 2016)和pTransE(Lin et al. 2015a)。另一方面，近期提出了多种双线性模型，比如DistMult(Yang et al. 2014)、HolE(Nickel, Rosasco, and Poggio 2016)和ComplEx(Trouillon et al. 2016)；它们都在link prediction task中的$HITS@\it{1}$上达到了高的准确度。TransE不能在$HITS@\it{1}$上产生比较好的结果，但是TransE能够在$HITS@\it{10}$上和双线性模型一争高下。我们发现产生TransE这种结果的原因是因为它的正则化。在嵌入向量空间中。TransE强制将实体嵌入到了一个球面上。它与TransE的原理相冲突（**这里理解为TransE将实体嵌入到球面上，但是TransE的原理是线性的**）并且扭曲了TransE获得的嵌入表征。这样使得TransE对link predictions的准确度产生不利影响，而TransE又非常需要它，因为嵌入在没有它的情况下会无限发散（diverge unlimitedly）。

本文，我们提出了一种模型通过将实体和关系嵌入到另一个嵌入空间（一个圆环，a torus）中的同时不需要任何正则化，但是和TransE有一样的原理。嵌入空间在TransE的策略下运行需要几个特征。该策略下的模型实际上能在李群的数学对象上很好的定义。通过选在一个紧凑的礼券作为嵌入空间，嵌入不会无限制的发散（diverge unlimitedly）并且不再需要正则化（regularization is no longer required）。因此，对于嵌入空间我们选择一个环，一种紧凑的李群，并提出了一个新颖的模型，TorusE。这个方法允许使用TransE相同原理更准确的学习嵌入表征，并且胜过所有可供选择的link prediction task方法。而且，TorusE可以很好的扩展到大规模的知识图谱上，因为相比于其他的方法它的复杂性是最低的，并且我们展示了TorusE比TransE更快，因为TorusE不需要在计算正则化了。

本文其余部分安排如下：在第二节，我们阐述了link prediction task的相关工作。在第三节，我们简要的介绍了最初的基于翻译的方法-TransE，并且涉及正则化的缺陷。然后，分析了从一个嵌入空间找另外一个嵌入空间所需要的条件（这句话感觉翻译得有点问题，附上原文：the conditions required for an embedding space are analyzed to find another embedding space）。在第四节，我们提出通过将一个空间改变为一个环从而获得嵌入表征的方法。这种方法客服了TransE中的正则化缺陷。在第五节，我们展示了我们做的实验，这个实验是将我们的方法和基准数据集的基准结果进行了比较。在第六节，我们总结了本论文。

## 2. 相关工作

通过link prediction task多种模型都提出了对知识图谱的补全。这些模型大致能分为3类：基于翻译的模型、双线性模型和基于神经网络模型。首先我们描述符号以方便下面的讨论。$h$、$r$和$t$分布表示为一个头实体（a head entity）、关系和一个尾实体（a tail entity）。加粗的字母$\boldsymbol{h}$、$\boldsymbol{r}$和$\boldsymbol{t}$分别表示$h$、$r$和$t$在嵌入空间$\mathbb{R}^n$的嵌入表征。$E$和$R$分别表示为一组实体和关系。

|编号|英语|词性|本文中翻译的中文|理解|
|---|---|---|---|---|
||suitable|adj|合适的|/|
||project|v|投影|这里做动词用|
||yield|v|出产(作物);产生(收益、效益等);提供;屈服;让步;放弃;缴出|/|
||restrict|vt|(以法规)限制;限定(数量、范围等);束缚;妨碍;阻碍;约束|这里用“约束”来翻译比较合适|
||eliminate|v|排除;清除;消除;(比赛中)淘汰;消灭，干掉(尤指敌人或对手)|/|
||redundancy|n|冗余;多余;(因劳动力过剩而造成的)裁员，解雇;累赘|/|
||conjugate|adj|共轭的|/|
||adequately|adv|充分地；足够地；适当地|/|
||distinguish|v|区分|/|

### 2.1 基于翻译的模型

TransE是第一个基于翻译的模型。它因为有效性和简洁性而备受关注。TransE受到了skip-gram模型（Mikolov et al. 2013a; 2013b）的启发，其中，单词嵌入的差异常被表示为它们的关系。因此，TransE使用$\boldsymbol{h}+\boldsymbol{r}=\boldsymbol{t}$原理。这原理能够有效地捕捉第一阶（first-order）规则，比如“$\forall e_1,e_2 \in E, \, (e_1,r_1,e_2) \rightarrow (e_1,r_2,e_2)$”、“$\forall e_1,e_2 \in E, \, (e_1,r_1,e_2) \rightarrow (e_2,r_2,e_1)$”和“$\forall e_1,e_2 \in E, \{\exists e_3 \in E,(e_1,r_1,e_3) \wedge (e_3,r_2,e_2)\}\rightarrow (e_2,r_3,e_1)$”。第一个是通过优化嵌入捕获的，以便$\boldsymbol{r}_1=\boldsymbol{r}_2$保持不变；第二个通过优化嵌入捕获的，以便$\boldsymbol{r}_1=-\boldsymbol{r}_2$保持不变；并且第三个通过优化嵌入捕获的，以便$\boldsymbol{r}_1 + \boldsymbol{r}_3 = \boldsymbol{r}_2$保持不变（这一段没有翻译好，附上原文：The first one is captured by optimizing embeddings so that r1 = r2 holds,the second one is captured by optimizing embeddings so that r1 = −r2 holds, and the third one is captured by optimizing embeddings so that r1 + r3 = r2 holds）。很多研究人员指出：该原理不适合表现$1-N,N-1 \;\text{and }\; N-N$关系。已经开发了一些TransE的扩展模型来解决这些问题。

TransH（Wang et al. 2014）将实体投影到对应于他们之间关系的超平面（hyperplane）上。投影（projection）通过选在嵌入表征的成分来表示他们之间的关系使得模型变得更具灵活性。TransR（Lin et al. 2015b）对每个关系都有一个矩阵，并且实体都通过线性转换（linear transformation）来实现映射，线性转换将矩阵相乘以计算三元组的分数（score）。TransR被认为是广义的TransH
，因为投影就是一种线性转换。这些模型相比于TransE都具有表达能力上的优势，但是，与此同时，它们很容易变得过拟合。

TransE能够在其他方面得到扩展。在TransG（Xiao, Huang, and Zhu 2016）中，在一个知识图谱中的关系能拥有多重含义，因此，通过多个向量来表示一个关系。PTransE(Lin et al. 2015a)将实体之间的关系路径考虑在内，以计算三元组的得分。关系路径由路径中每个关系的总和来表示。

### 2.2 双线性模型

近期，双线性模型在link prediction方面取得了很好的结果。RESCAL（Nickel,Tresp, and Kriegel 2011）是第一个双线性模型。每个关系通过一个$n \times n$的矩阵来表示，并且三元组$(h,r,t)$的分数通过一个双线性映射来计算，该映射对应关系r的矩阵和它的参数$\boldsymbol{h}$、$\boldsymbol{t}$。因此，RESCAL也是最广义的双线性模型。

通过约束双线性函数提出了多个RESCAL扩展模型。DistMult（Yang et al.2014）将表示关系的矩阵限制为对角矩阵。DistMult是的模型容易被训练并且消除了冗余（eliminates the redundancy）。尽管，它也存在$(h,r,t)$和$(t,r,n)$分数一样的问题。为了解决这个问题，ComplEx（Trouillon et al. 2016）使用复数代替实数，并且，在计算双线性映射之前采用尾实体嵌入表征的共轭嵌入。三元组的分数是双线性映射输出的复数的实数部分。

双线性模型相比于基于翻译的模型有很多冗余，并且更容易过拟合。因此，嵌入空间被限制在了一个低维空间中。这也可能导致其在大规模知识图谱上使用时产生问题，大规模的知识图谱是指包含大量实体的知识图谱。因为，需要高维空间来嵌入实体，因为只有高维空间才能充分的区分这些嵌入的实体。

### 2.3 基于神经网络的模型

基于神经网络的模型类似于神经网络的结构，它有很多层和一个激活函数。神经张量网络（neural tensor network, NTN）（Socher et al. 2013）有一个标准的线性神经网络结构和一个双线性张量结构。它可以认为是一个广义的RESCAL。通过每一个关系来训练网络的权重。EP-MLP（Dong et al.2014）是一个NTN的简化版本。

基于神经网络的模型是三类模型中最具表现力的模型，因为它们有大量的参数。因此，它们可能捕获很多类型的关系，但是，与此同时，它们也最容易对训练数据过拟合。

## 3. TransE和它的缺陷

在本节中，我们将阐述TransE的细节，并且展示它的正则化缺陷。在本文的后半部分，我们提出一个和TransE相似策略的新颖的模型来克服以上缺陷。

TransE算法包含以下三个主要部分：

1. 原理：当$\Delta \text{是一组真实的三元组时}, if\;(h,r,t) \in \Delta, \text{保持} \boldsymbol{h}+\boldsymbol{r}=\boldsymbol{t} \text{不变}$TransE学习嵌入表征。为了测量有多少个三元组嵌入遵循该原理，使用了一个记分函数$\it{f}$。通常$\boldsymbol{h}+\boldsymbol{r}-\boldsymbol{t}$的$L_1$范数或者$L_2$范数的平方记为$\it{f}(h,r,t)$。在这种情况下$\it{f}(h,r,t)=0$意味着$\boldsymbol{h}+\boldsymbol{r}=\boldsymbol{t}$能一直保持（holds completely）。
2. 负采样：如果只有原理，那么TransE只是学习到了一个trivial的解决方案，这个解决方案中所有的实体嵌入都是相同的并且所有的关系嵌入都是0。因此，负采样是必须的。通常一个知识图谱只包含正向三元组（也就是正样本），因此TransE通过随机修改每个真实三元组的后或者尾实体来生成一个负三元组（负样本）。这被称为负采样。TransE学习嵌入表征使得$\it{f}(h',r,t,)$变大，$\text{if}(h',r,t')\in \Delta_{h,r,t}',\text{其中}(h,r,t)\in \Delta \;\text{并且}\; \Delta_{h,r,t}'=\{(h',r,t)|h'\in E, h' \not =h\}\cup \{(h,r,t')|t'\in E, t'\not= t\}$。
3. 正则化：为了不让嵌入表征无限制的分散，正则化是必须的。TransE使用归一化（normalization）作为正则（regularization）。将实体的嵌入表征归一化，以便它们的大小（magnitude）在学习step中变为1。也就是说，对于每个实体$e\in E,\boldsymbol{e}\in S^{n-1}\subset \mathbb{R}^n ,\, \text{其中}S^{n-1}\text{是一个维度为n-1}的球$。

TransE使用[margin loss](https://zhuanlan.zhihu.com/p/101143469)。目标函数定义如下：
$$\mathcal{L}=\sum\limits_{(h,r,t)\in \Delta}\sum\limits_{(h',r,t')\in \Delta_{(h,r,t)}'}[\gamma + \it{f}(h,r,t)-\it{f}(h',r,t')]_+ \tag{1}$$
其中$[x]_+$表示x的正向部分，并且$\gamma>0$是一个margin超参数。TransE通过随机梯度下降来训练。

如果实体和关系都被键入到一个真实的向量空间中，三个部分都是必要的。尽管原理和正则化在训练期间是互相矛盾的，因此对于每个$e\in E \text{并且}r \in R, \, \boldsymbol{e}+\boldsymbol{R} \notin S^{n-1}$几乎一直保持不变。因此，在大多数环境下，很少实现原理$\boldsymbol{h}+\boldsymbol{r}=\boldsymbol{t}$，如图1所示。![当$n$等于2时TransE获取嵌入图。它假设$(A,r,A'),(B,r,B')$和$(C,r,C')$保持不变](../pictures/TorusE_Figure1.png "当$n$等于2时TransE获取嵌入图。它假设$(A,r,A'),(B,r,B')$和$(C,r,C')$保持不变")


|编号|英语|词性|本文中翻译的中文|理解|
|---|---|---|---|---|
||simplify|vt|简化;使简易|/|
||trivial|adj|不重要的;琐碎的;微不足道的|/|
||magnitude|n|巨大;重大;重要性;星等;星的亮度;震级|这里翻译为大小|
||that is|/|也就是说，即|/|
||conflict|v/n|冲突/矛盾|/|
|||||/|
|||||/|
|||||/|
|||||/|
|||||/|
|||||/|
|||||/|
