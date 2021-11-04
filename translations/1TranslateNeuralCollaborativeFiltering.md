# translate翻译

1. 题目：neural collaborative filtering
神经协同过滤

2. 翻译说明：

|number|title of paper|internet source|local source|correlative field|illustration|
|---|---|---|---|---|---|
|1|neural collaborative filtering|<http://staff.ustc.edu.cn/~hexn/papers/www17-ncf.pdf>|/references/1NeuralCollaborativeFiltering.pdf|recommonder system|English translate into chinese|

## abstract 摘要

|编号|英语|本文中翻译的中文|理解|
|---|---|---|---|
|1|immense|巨大的|/|
|2|speech recognition|语音识别|/|
|3|Neural network-based Collaborative Filtering|NCF基于协同过滤的神经网络|/|
|4|arbitrary function|任意函数|可能不是特指什么函数，而是泛指所有函数。|
|5|implicit feedback|隐性反馈|显式反馈就是用户对item的打分。例如按照评分1-5来打分，不同的打分就代表用户对item的不同喜好程度。隐性反馈不是打分，可以看做是一种选择。用户选择了某个item。举例来说这个选择可以是社交网络中的点赞、转发等等。除了用户所选择的item之外，剩下的item都是未选择的，而未选择不代表用户不喜欢这些item。有可能是因为还未看到这些item等原因。这就是隐性反馈与显示反馈的区别。例如Gowalla这样的数据，就是隐性反馈数据。[参考](https://blog.csdn.net/yinruiyang94/article/details/78906370)|
|6|state of the art|传说中的SOTA|重大改进和显著提高|
|7|matrix factorization, MF|矩阵分解|把矩阵A分解为矩阵U和V的乘积。[参考](https://blog.csdn.net/u014595019/article/details/80586438)|
|8|empirical evidence|实验证据表明|有些地方把它翻译为“经验表明”，这里理解为实验表明。|


在最近几年，深度神经网络已经在语音识别、计算机视觉和自然语言处理领域取得的巨大的成功。然而，深度神经网络在推荐系统的探索较少的受到关注（scrutiny）。在这个工作中，我们努力开发了基于神经网络的技术来处理推荐系统中的关键问题——协同过滤——在隐性反馈（implicit feedback）的基础上。  
尽管最近有些工作已经在推荐系统中使用了深度学习，他们主要使用它去模拟辅助信息，诸如对象（item）的文字描述和音乐声纹特征。在分析用户和对象特征之间的交互作用时，虽然深度学习已经成为协同过滤模型的关键因素，它们仍然使用（resort）矩阵分解的方法，并对用户和对象的隐藏特征进行内积操作。  
通过将内积替换为一个可以从数据中学习任意函数（arbitrary function）的神经结构，我们提出了一个名为：NCF的通用框架，NCF是基于神经网络的协同过滤的简称。NCF是通用的，并且再它的框架下NCF能表达、推广（express and generalize）矩阵分解。为了增强NCF模型在非线性情况下的表现，我们计划借助（leverage）多层感知机来学习用户项(item)交互函数。在两个真实数据集上的大量实验表明，与最先进的方法相比（state of the art）我们提出的NCF框架有了显著的提高（significant improvements）。实验证据表明（empirical evidence）使用更深层的神经网络可以提供更好的推荐性能。  

关键词
协同过滤，神经网络，深度学习，矩阵分解，隐性反馈。

## 1. 介绍

|编号|英语|本文中翻译的中文|理解|
|---|---|---|---|
|1|project|映射，投影|应该表达了空间映射的关系|
|2|shared latent space|共享的隐性空间|可能是将两种数据共同映射到了一个使它们可以发生联系的空间中，是不是在暗示对**矩阵进行线性变化之后对应到一个可能线性相关的空间中了**。|
|3|represent|表示、代表|注意后面的数量形式，加了量词a和an。|
|4|inner product|内积|矩阵乘法|
|5|de facto|事实上的|/|
|6|approach|方法，名词|这里表示的是名词方法，不是动词。|
|7|model-based|模型|是一个专有名词，有对比的两个词组model-free和model-based，是两种方法。model-based可以理解为人工提取特征的，也就是特征在模型运算过程中不能更改。而对应的神经网络是“学习”特征，这种特征是隐性的、非人工指定的。[参考1](https://zhidao.baidu.com/question/694026622784498004.html)[参考2](https://www.zhihu.com/question/64369408)。**还是没有理解这个位置**。|
|8|factorization machines|因子分解机|[参考](https://blog.csdn.net/lijingru1/article/details/88623136)|
|9|explicit feedback|显示反馈|/|
|10|bais term|偏置项|[参考](https://www.codenong.com/cs109787538/)|
|11|interaction function|交互函数|通过偏置项这里的说明可以推断这个是交互函数，而不是交互功能。|
|12|latent|隐性|这里统一一下，应该理解为“隐性”这种翻译，而不是“潜在”这种翻译。|

在信息爆炸的时代，推荐系统在缓解信息过载扮演着关键作用，而且它已经在众多在线服务中广泛使用，比如电子商务、在线新闻和社交网站。个性化的推荐系统关键在于根据用户过去与对象（item）之间的互动（比如评分和点击）而表现出来的偏好进行建模，成为协同过滤。在各种协同过滤技术中，矩阵分解（MF）是最流行的一种，他将用户和对象的映射到一个共享隐性空间中，使用隐性特征向量来代表**一个**用户和**一个**对象。以后，用户和对象（item）的交互被做成它们隐性向量的内积模型。

通过Netfix奖的推广，矩阵分解已经成为事实上方法，~~这种方法是隐性因素model-based推荐系统的~~。许多研究工作致力于增强矩阵分解，例如将其MF和基于邻居（neighbor-based）的模型集成、将MF与item内容的主题模型（topic models）组合起来，并将MF扩展到分解机（factorization machines），以便于对通用的特征进行建模。尽管对于协同过滤而言MF是有效的，但众所周知，交互函数的内积（interaction function inner product）的简单选择（simple choice）会阻碍MF的性能。例如，对于显示反馈中的评分预测任务，大家都知道，MF模型的性能能通过将用户和item的偏置项包含到交互函数中得到提高。虽然它仅仅似乎只是对内积运算（inner product operator）一个细微的调整，但它指出了（points）对于如何设计更好、更专业的交互函数来模拟用户们和items之间的隐性特征交互达到更好的效果。内积只是简单线性组合隐性特征的乘法，它可能不能够很好的获取（capture，这里应该是获取或者体现的意思）用户交互数据的复杂结构。

|编号|英语|中文|理解|
|---|---|---|---|
|1|approximating|拟合|对一个连续函数的无限逼近。|
|2|deep neural network, DNN|深度神经网络|/|
|3|ranging from|从...到什么的排列|主要就是列举一些例子用的。|
|4|natural scarcity|天然稀缺性|/|
|5|perceptron|感知机|/|

本文探究了如何使用深度神经网络从数据中学习（自动学习，也就是传说中的end to end）得到交互函数的问题，而不是以前的通过手工（人工）来完成交换函数设置的工作。神经网络已经被证明具备拟合（approximating）任何连续函数的能力，并且最近发现深度神经网络在多个领域都是有效的，比如计算机视觉、语音识别到文本处理。然而，相比于MF方法的大量文献，在推荐领域应用（employing）DNN的相关工作很少。尽管一些近期的进展已经将DNNs应用到推荐任务中并且显示出了承诺（promising）的结果，它们大多使用DNNs对辅助信息（auxiliary information）建模，诸如，items的文本描述、音乐的音频特征和图像的视觉内容。我们注意到对于关键协同过滤效果（key collaborative filtering effect）的建模，他们仍然采用（resorted）MF方法，这个MF使用一个内积来将用户和item隐性特征组合在一起。

本文工作通过将一个完成协同过滤功能的神经网络建模方法形式化（formalizing）来解决（addresses）上述研究问题（也就是通过实现一个完成协同过滤功能的神经网络来解决上述问题）。我们关注于隐性反馈，隐性反馈间接的通过观看视频、购买产品和点击items来反馈用户的偏好。与显性反馈相比（评级和评论），隐性反馈可以被自动跟踪，并且内容提供商更容易去收集这些信息。然而，隐性反馈的使用更具有挑战性，这是因为用户的满意度并没有被直接观察到，并且负面反馈在隐性反馈中具有天然的稀缺性。在本文中，我们研究的主题（central theme）是如何使用DNNs对含有噪声的隐性反馈信号建模。

这项工作的主要贡献如下：

1. 我们对于用户们和items的隐性特征模型提出了一种神经网络结构，并且对于协同过滤设计了一种基于神经网络的通用框架NCF。
2. 我们明确提出，MF能被解释为NCF的一种特殊形式，并且使用多层感知机(perceptron)将高级非线性性赋予了NCF模型。
3. 我们使用两个真实数据集来执行大量的实验来论证（demonstrate）我们NCF方法的有效性（effectiveness）和对协同过滤使用深度学习的前景（promise）。

## 2. 准备工作（preliminaries）

我们首先将问题进行了形式化处理，并且讨论现有隐性反馈的协同过滤解决方案。然后我们简要的概述了广泛使用的MF模型，凸显（highlighting）了他因为使用内积而造成的局限性（limitation）。

### 2.1 从隐性数据学习

**github上无法正常显示LaTex的公式，使用Chrome的GitHub Math Display即可正常显示。另外说明一点MathJax Plugin for Github插件并不好用，不能正确显示下文中的多行公式。**  

设$M$和$N$分别表示用户们和物品的数量。我们从用户的隐性反馈中定义user-item交互矩阵为$Y\in R^{M\times N}$。如（这个Y我理解上就是数据的标签。）
$$y_{ui}=\begin{cases}
1 &\text{if interaction(user u, item i)is  observed;}\\
0 &\text{otherwise.}
\end{cases} \tag{1}
$$

这里的$y_{ui}$值为1的时候表示user $u$和item $i$之间存在一次交互；然而这并不意味着u真的喜欢i。类似的，值为0页并不代表u不喜欢i，值为0可能表示user不知道item的存在。这也导致了从隐性数据进行学习的挑战，因为隐性数据只提供了用户偏好的噪声信号（only noisy signals about users' preference）。虽然观察到来的条目（entries）至少反映了用户对该item感兴趣，但是没有观察到的条目有可能只是数据缺失，并且负面反馈存在天然稀缺性。

|编号|英语|中文|理解|
|---|---|---|---|
|1|underlying model|潜在模型或者基础模型|还不太理解|
|2|pointwise loss|单点损失|pointwise和pairwise都是对物品偏好程度的平方方法。单点主要是对单一物品评分的拟合，注重对评分的拟合程度。|
|3|pairwise loss|双点损失|而pairwise可能的操作方式是先选择一些没有被购买的item作为负样本，然后将已经购买的作为正样本。这样计算两种样本之间的差值。也就是说pairwise更关注的是pair样本之间的关系。|
|4|listwise loss|列表损失|本文中目前还没有出现这种loss，这里只是做一点扩展。|

在隐性反馈的推荐系统中存在的问题被描述为估计（estimation）$Y$中未观察到的条目的分值（scores）的问题，这个分数用于对条目进行排序。model-based方法假定数据能够通过一个深层模型（underlying model）来生成（或者被描述）。形式上，他们能被$\hat y_{ui} = f(u,i|\Theta)$，其中$\hat y_{ui}$表示交互$y_{ui}$的预测分数（pridicted score），$\Theta$表示为模型的多个参数（model parameters），$f$表示将模型多个参数映射到预测值的的函数（我们将其称为一个交互函数（an interaction function））。

为了估计参数$\Theta$，现有的方法遵循优化一个目标函数的机器学习范式。文献中常用目标函数有两种：pointwise loss和pairwise loss。作为显性反馈评估的自然延伸，pointwise learning通常按照回归框架，通过求$\hat y_{ui}$和目标值$y_{ui}$之间的平方损失（squared loss）的最小值来实现。为了处理负面评价数据的缺失（absence of negative data），他们要么将所有未观察到的entries视为负反馈，要么从未观察到的entries中进行抽样负面实例（negative instances）作为负反馈（这句话的意思就是如何提取负反馈的方法，二选一：要么将所有的非正反馈都视作负反馈，要么从所有的非正样本中抽样一部分作为负样本）。对于pairwise learning而言，其出发点是观察到的条目的评级应该比未观察到的条目搞。因此，pairwise learning通过求观察到的条目的$\hat y_{ui}$和未观察到条目的$\hat y_{ui}$之间的最大幅值（maximizes the margin）来代替求$\hat y_{ui}$和$y_{ui}$之间的最小化损失。

我们向前进了一步，我们的NCF框架使用神经网络参数化交互函数$f$来估计$\hat y_{ui}$的。因此，NCF框架天然支持pointwise和pairwise learning。

### 2.2 矩阵分解

MF把每个user和item的隐性特征真实值向量相互关联。设$p_u$和$q_i$分别表示user u和item i的隐性向量；MF将预估一个交互$y_{ui}$为$p_u$和$q_i$的内积。
$$\hat y_{ui}=f(u,i|p_u, q_i) = p_u^T q_i=\sum \limits_{k=1}^K p_{uk}q_{ik}, \tag{2}$$  
当$K$表示隐性空间的维度（dimension）。如我们所见，MF对user和item之间的双向交互（two-way interaction）进行了建模，假设隐性空间的每个维度之间彼此独立，并且以相同的权重线性组合。因此MF可视为隐性因素的线性模型。

|编号|英语|中文|理解|
|---|---|---|---|
|1|ground truth similarity|真实值。一词指的是训练集对监督学习技术的分类的准确性。这在统计模型中被用来证明或否定研究假设。|[参考](https://blog.csdn.net/qq_15150903/article/details/84789591)|
|2|jaccard coefficient|Jaccard相似系数。用于比较有限样本集之间的相似性与差异性。**Jaccard系数值越大，样本相似度越高**。|[参考](https://baike.baidu.com/item/Jaccard%E7%B3%BB%E6%95%B0/6784913?fr=aladdin)|
|3|note|注意到|这里翻译为注意到比较合适。如果翻译为指出，好像和后面有点对不上。|

![avatar](/pictures/1TranslateNeuralCollaborativeFiltering_Figure1.png)  
图1：说明MF局限性的一个例子。根据数据矩阵（data matrix）(a)$u_1$和$u_4$最相似，其次值$u_3$，最后是$u_2$。然而，在隐性空间b中，将$p_4$放在距离$p_1$最近的位置会使得$p_4$比$p_3$更靠近$p_2$，从而导致巨大的排名损失。

图1说明了内积函数（inner product function）如何限制MF的表达能力的。为了更好的理解示例，有两种设置需要事先明确说明。首先，由于MF将users和items映射到了同一个隐性空间，因此两个用户之间的相关性（similarity）也可以通过内积或者等式(2)来度量，内积是两个用户隐性向量之间夹角的余弦。第二，在不丧失一般性的情况下，我们使用**jaccard系数作为衡量MF需要恢复的两个用户之间真实相似程度的评判标准**。

$$
\text{Use Jaccard coefficient to calculate similarity} \\
\text{given two sets A and B, Jaccard coefficient is defined as follows :} \\
J(A,B) = \frac{A\cap B}{A \cup B} = \frac{A\cap B}{|A| + |B| - |A\cap B|}
$$

让我们首先聚焦于图1a的前3行。容易得到$s_{23}(0.66)>s_{12}(0.5)>s_{13}(0.4)$。同样的，$p_1$、$p_2$和$p_3$在隐性空间之间的几何关系能被绘制如图1b所示。现在，让我们考虑一个新user $u_4$，其输入如图1a中虚线所示。我们能得到$s_{41}(0.6)>s_{43}(0.4)>s_{42}(0.2)$，这意味着$u_4$和$u_1$最相似，其次和$u_3$，最后是$u_2$。然而，如果一个MF模型将$p_4$放在最靠近$p_1$的位置（如图1b中两个虚线显示了2个选择），这将导致$p_4$将比$p_3$更靠近$p_2$，不幸的是这将导致巨大的ranking loss。

在上面的例子中显示了MF可能的局限性，这种局限性是由于在低维隐性空间（low-dimensional latent space）中使用简单和固定的内积来估计复杂的user-item交互关系导致的。我们注意到（note）一种途径来解决这个问题。这个途径是使用大数据量的隐性特征$K$（这是不是意味着其实并没有从方法上来解决这个问题，而是通过巨大的数据量来冲淡了这个问题的影响？）。尽管这种方法可能会对模型通用性（generalization）产生不利影响（比如：对特定数据过拟合（overfitting the data）），特别是在稀疏设置中（可能是在稀疏的数据的情况下）。在本工作中，**我们通过从数据使用DNNs学习交互函数来应对该局限性**。

## 3. 神经协同过滤

|编号|英语|中文|理解|
|---|---|---|---|
|1|binary property|二值化属性|还不理解是什么意思。|
|2|margin-based loss|基于边缘的损失|和后面的ranking loss好像是同一个事物的不同名字或者表现。[参考，这里面说了和ranking loss的关系，说margin loss只是ranking loss的另外一种表现形式](https://zhuanlan.zhihu.com/p/158853633)|
|3|bayesian personalized ranking|rangking loss，在训练集上使用ranking loss函数是非常灵活的，我们只需要一个可以衡量数据点之间的相似度度量就可以使用这个损失函数了。这个度量可以是二值的（相似/不相似）。比如，在一个人脸验证数据集上，我们可以度量某个两张脸是否属于同一个人（相似）或者不属于同一个人（不相似）。这样就和前面的Jaccard系数联系起来了。注意这里使用的是pairwise loss。|[参考](https://zhuanlan.zhihu.com/p/158853633)|

我们首先提出通用NCF框架，详细阐述了如何使用概率模型来获得（learn）NCF，其中概率模型强调隐性数据的二进制属性（that emphasizes the binary property of implicit data）。随后我们证明了MF可以用NCF表示和概括（expressed and generalized）。为了研究用于协同过滤的DNN，随之我们列举了一个NCF的实例，使用多层感知器（a multi-layer perceptron）来学习user-item的交互函数。最后，我们提出一个新的神经矩阵分解模型，该模型在NCF框架下集成了MF和MLP；它结合了MF的线性和MLP的非线性的优点来对user-item隐性结构建模。

### 3.1 通用框架

![avatar](/pictures/1TranslateNeuralCollaborativeFiltering_Figure2.png)  

为了得到一个全神经化的协同过滤方法，我们采用多层结构（multilayer representation）来对user-item交互$y_ui$进行建模，如图2所示，其中一层的输出作为下一层的输入。底部的输入层由两个特征向量$v_u^U$和$v_i^I$组层，它们分别描述了user $u$和item $i$，为了支持各种各样的users和items类型，可以对user $u$和item $i$进行改造（can be customized）以适应各种应用场景。诸如上下文感知（context-aware）、基于内容的方法（content-based）、基于邻居的方法（neighbor-based）。由于本工作的重点在于单纯的协同过滤环境（collaborative filtering setting），因此我们仅使user和item的标识（identity）作为输出特征，通过一个独热编码（one-hot encoding）将其转换为一个二值化的稀疏向量。需要注意的是，~~使用这种通用特征标识，~~（这句话感觉和后面的重复了）我们的方法通过使用内容特征来表示user和item，能够非常容易的调整以解决冷启动的问题。

在输入层之上是嵌入层（embedding layer），它是一个全连接层，它将稀疏表示的向量映射到一个稠密向量上。所得到的user(item)嵌入向量可以视为在隐性特征模型语境（in the context of latent factor model）中的user(item)隐性向量。然后，user embedding和item embedding将会输入到一个多层神经体系中，以上我们称其为神经协同过滤层。神经协同过滤层的作用就是用隐性向量来预测分数（to map the latent vectors to prediction scores）。神经协同过滤层（神经同过滤层包含有多层）的每一层都可以进行调整，用以发现user-item交互之中确定的隐性结构（certain latent structures of user-item interactions）。最后的隐藏层$X$的维度决定了模型的能力。最终输出层用于预测分数$\hat y_{ui}$，通过计算$\hat y_{ui}$和$y_{ui}$之间的pointwise loss最小化作为目标函数来进行训练。我们注意到有另外的途径使用pairwise learning来训练模型，这种途径诸如bayesian personalized ranking和margin-based loss。由于本文的重点是神经网络建模部分，所以我们将不会展开讨论NCF在pairwise learning上的情况（we leave the extension to pairwise learning of NCF as a future work）。

|编号|英语|中文|理解|
|---|---|---|---|
|||||
|||||
|||||
|||||

我们将NCF预测模型形式化的定义为：
$$\hat y_{ui}=f(P^T v_u^U, Q^T v_i^I|P,Q,\Theta_f) \tag{3}$$
其中$P\in R^{M \times K}$且$Q\in R^{N \times K}$，分别表示了user和item的隐性特征矩阵；$\Theta_f$表示了模型交互函数$f$的模型参数。由于函数$f$被定义为一个多层神经网络，因此$f$可以形式化定义为：
$$f(P^T v_u^U, Q^T v_i^I)=\phi_{out}(\phi_X(...\phi_2(\phi_1(P^T v_u^U, Q^T v_i^I))...)) \tag{4}$$
