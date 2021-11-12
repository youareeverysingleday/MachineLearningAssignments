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
|9|item|物品|对不同的环境特指的是不同与用户交互的物品。比如在电商领域，这就指的是商品。一般性的翻译为物品。|

在最近几年，深度神经网络已经在语音识别、计算机视觉和自然语言处理领域取得的巨大的成功。然而，深度神经网络在推荐系统的探索较少的受到关注（scrutiny）。在这个工作中，我们努力开发了基于神经网络的技术来处理推荐系统中的关键问题——协同过滤——在隐性反馈（implicit feedback）的基础上。  
尽管最近有些工作已经在推荐系统中使用了深度学习，他们主要使用它去模拟辅助信息，诸如物品（item）的文字描述和音乐声纹特征。在分析用户和物品特征之间的交互作用时，虽然深度学习已经成为协同过滤模型的关键因素，它们仍然使用（resort）矩阵分解的方法，并对用户和物品的隐藏特征进行内积操作。  
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

在信息爆炸的时代，推荐系统在缓解信息过载扮演着关键作用，而且它已经在众多在线服务中广泛使用，比如电子商务、在线新闻和社交网站。个性化的推荐系统关键在于根据用户过去与物品（item）之间的互动（比如评分和点击）而表现出来的偏好进行建模，成为协同过滤。在各种协同过滤技术中，矩阵分解（MF）是最流行的一种，他将用户和物品的映射到一个共享隐性空间中，使用隐性特征向量来代表**一个**用户和**一个**物品。以后，用户和物品（item）的交互被做成它们隐性向量的内积模型。

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

我们将NCF预测模型形式化的定义为：
$$\hat y_{ui}=f(P^T v_u^U, Q^T v_i^I|P,Q,\Theta_f) \tag{3}$$
其中$P\in R^{M \times K}$且$Q\in R^{N \times K}$，分别表示了user和item的隐性特征矩阵；$\Theta_f$表示了模型交互函数$f$的模型参数。由于函数$f$被定义为一个多层神经网络，因此$f$可以形式化定义为：
$$f(P^T v_u^U, Q^T v_i^I)=\phi_{out}(\phi_X(...\phi_2(\phi_1(P^T v_u^U, Q^T v_i^I))...)) \tag{4}$$
其中$\phi_{out}$和$\phi_X$分别表示输出层和第x个神经协同过滤（CF）层的映射函数，这其中共有x个神经协同过滤层。

#### 3.1.1 学习NCF

为了学习模型参数，现有的pointwise方法主要使用平方损失进行的回归方法。
$$L_{sqr}=\sum\limits_{(u,i)\in y \cup y^-} w_{ui}(y_{ui}-\hat{y}_{ui})^2 \tag{5}$$  
其中$y$表示为在$Y$中的一组可见交互，$y^-$表示为一组负面实例，$y^-$可以是不可见交互的全部或者从其中抽样一部分；$w_{ui}$是一个超参数，用于表示训练实例$(u,i)$的权重。虽然squared loss能通过假定可见值符合高斯分布，但是我们指出，这种分布规律可能并不能与隐性数据的真实规律符合得很好（就是高斯分布和隐性数据的真实分布并不一致）。这是因为对于隐性数据，目标值$y_{ui}$是二值化的1和0，用以表示u和i是否交互。在下文中，我们提出了一种学习pointwise NCF的概率性的方法，该方法对隐性数据的二进制性质给予了特别的关注。  

|编号|英语|中文|理解|
|---|---|---|---|
|1|one-class nature|不清楚如何翻译|/|
|2|probabilistic function|概率函数|/|
|3|likelihood function|似然函数|[参考](https://blog.csdn.net/caimouse/article/details/60142085)|
|4|cross entropy loss|交叉熵|/|
|5|log loss|对数损失|[参考](https://blog.csdn.net/laolu1573/article/details/82925747)[对数损失和交叉熵是一个东西](https://zhuanlan.zhihu.com/p/96798110)[对数损失的详细说明](http://neuralnetworksanddeeplearning.com/chap3.html#the_cross-entropy_cost_function)[各种损失函数说明](https://zhuanlan.zhihu.com/p/44216830)|
|6|item popularity biased|物品受欢迎的偏好|对于非均匀抽样进行的一种策略，可还没有具体的查看文献14和12，猜测可能对有偏好的物品进行更多的抽样。|

考虑到隐性反馈的one-class nature，我们可以将$y_{ui}$的值当做是一个标签：1表示item i和u有关，否则为0。预测分值$\hat y_{ui}$表示（represents）i和u相关的可能性（how likely i is relevant to u）。为了赋予NCF的输出这种概率解释，我们需要将输出值$\hat y_{ui}$的范围限制在$[0,1]$之间，这样我们就可以使用概率函数（probabilistic function）（诸如 logistic or probit function）来轻松实现输出层$\phi_{out}$的激活函数。通过上述设置，我们将似然函数定义如下：  
$$p(y,y^-|P,Q,\Theta_f)=\prod_{(u,i)\in y}\hat y_{ui} \prod_{(u,i)\in y^-}(1-\hat y_{ui}) \tag{6}$$  
取似然的负对数，我们得到（由于连乘不好计算，所以通过取对数将其转换为加法来进行计算）：  
$$L =-\sum \limits_{(u,i)\in y} \log \hat y_{ui}-\sum \limits_{(u,i)\in y^-}\log (1-\hat y_{ui})\\
 = -\sum\limits_{(u,i)\in y \cup y^-} y_{ui} \log \hat{y}_{ui} + (1-y_{ui}) \log{(1-\hat{y}_{ui})}\tag{7}$$  
这种NCF方法中需要将目标函数的最小化，并且使用随机梯度下降（stochastic gradient descent, SGN）可以对其优化。细心的读者可能已经意识到，它与二进制交叉熵损失（也称为对数损失）是相同的。通过对NCF进行probabilistic treatment，我们将隐性反馈推荐作为一个二元分类问题来处理。由于在推荐系统的相关文献中，很少对classification aware的对数损失（log loss）进行研究，我们这本项工作中对其进行了探讨，并在第4.3节中通过实验证明了使用对数损失的有效性。对于负面样本$y^-$我们的采样策略为：在每轮迭代中，我们均匀的（uniformly）从未观察到的交互中进行采样，并根据观察到的相互作用的数量控制采样率。~~并且控制采样率为$w.r.t$为可以观察到的交互数量。~~虽然非均匀的（nonuniform）采样策略（例如，item受欢迎程度偏好（popularity biased））可能会进一步的提高性能（是不是这样表述说明了这项工作中已经考虑了偏好问题？），但是我们将这一样工作留给未来（读到这句换，是不是意味着没有考虑偏好对模型的影响？）。

### 3.2 通用矩阵分解（GMF）

我们现在将说明MF能被解释为我们NCF框架的一种特殊形式。MF是推荐系统中最受欢迎的一种模型，并且MF已经在各种文献中得到了广泛的研究。如果NCF能够完整的模拟MF的能力，那么也就意味着NCF能够完整的模型基于MF的这一系列的因子分解模型（这句话的主语有点没读懂到底是MF还是NCF，按语法来说应该是MF，但是感觉怪怪的。这样的翻译是根据自己对上下文的理解来做出的。原文：being able to recover it allows NCF to mimic a large family of factorization models）。  

|编号|英语|中文|理解|
|---|---|---|---|
|1|identity function|恒等映射，恒等函数|对于任意输入$x$，都有$f(x)$使得$f(x)=x$，称为恒等映射或者恒等函数。[参考](https://blog.csdn.net/None_Pan/article/details/106394920)|

由于在输入层对user（item）都进行了one-hot编码，获得的embedding vector可以被视为user(item)的隐性向量。设user latent vector $p_u$为$P^T v_u^U$，item latent vector $q_i$为$Q^T v_i^I$。我们定义第一个神经CF层的映射函数如下：
$$\phi_1(p_u,q_i) = p_u \cdot q_i \tag{8}$$
将$\cdot$定义为向量element-wise之间的乘积（where $\cdot$ denotes the element-wise product of vectors.）。我们将上述向量投影到输出层上：  
$$\hat{y}_{ui} = a_{out}(h^T(p_u \cdot q_i)) \tag{9}$$
将$a_{out}$和$h$分别定义为输出层的激活函数和边上的权重。直观的，如果我们对于$a_{out}$使用一个idtentity function，并且让$h$全为1的统一向量（uniform vector，这里应该是说向量里面所有的元素都为1），我们就可以精准的呈现出MF模型。  
在NCF框架下，可以容易的对MF进行推广和扩展。例如，如果我们允许$h$能从数据中学习，并且突破统一的限制（uniform constratint），那么我们将得到一个MF的变种（a variant of MF），这个变种允许隐性空间维度重要性的差异化。如果我们对$a_{out}$使用一个非线性的函数，$a_{out}$将把MF扩展到非线性的情况下的MF，这可能是的非线性的MF比线性MF模型具备更好的表现。在本工作中，我们将在NCF框架下实现一个MF的广义版本（generalized version），这个版本中使用sigmoid function$\sigma(x) = 1/(1+e^{-x})$为$a_{out}$，使用从带有log loss的数据中学习得到$h$（section 3.1.1）。我们将这种形式的称之为generalized matrix factorization，缩写为GMF。

### 3.3 多层感知机（Multi-Layer Perceptron, MLP）

因为NCF使用了2种方法对user和item建模，直观的通过将两种方法连接起来达到组合两种方法特征的目的。这种设计已经被广泛的应选用在深度学习的多种模型中。尽管，简单的向量连接并不能解释user和item的潜在特征之间的任何交互，这样不足以为协同过滤进行建模。为了解决这个问题，我们建议在连接向量（concatenated vector）上添加隐藏层，使用标准的MLP来了解user和item之间的隐性特征的交互。为了学习到$p_u$和$p_i$之间的交互，我们需要赋予模型很大的灵活性和非线性，而不是只使用固定元素的GMF方法。更准确的说，我们NCF框架下的MLP模型定义为：
$$z_1 = \phi_1 (p_u, q_i) = \begin{bmatrix}
p_u \\
q_i \\
\end{bmatrix} \\
\phi_2 (z_1) = a_2 (W_2^T z_1 + b_2), \\
... \\
\phi_L (z_L -1) = a_L(W_L^T z_{L-1} + b_L), \\
\hat{y}_{ui} = \sigma (h^T \phi_L (z_L -1)), \tag{10}
$$

|编号|英语|中文|理解|
|---|---|---|---|
|1|hyperbolic tangent(tanh)|双曲正切|/|

其中$W_x$、$b_x$和$a_x$分别表示第x层感知机的矩阵权重、偏好向量和激活函数。对于MLP层的激活函数可以自由选择以下其中之一：sigmod、hyperbolic tangent(tanh)和Rectifier(ReLU)。我们将逐个分析每个函数：1)sigmoid函数会限制每个神经元处于(0,1)之间，这可能会限制模型的性能；众所周知，他会suffer from saturation，当神经元的输出接近于0或者1的时候，神经元就会停止学习。2)虽然，tanh是一个更好的选择，并且已经被广泛使用，但是它只是在某种程度上缓解了sigmod的问题，因为它可以是当做simgod$(tanh(x/2) = 2 \sigma(x)-1)$的重制版本。3)因此，我们选择ReLU，ReLU更合理（biologically plausible）并且被证明是非饱和的（non-saturated）；另外，它能够激励稀疏激活函数，因而非常适合稀疏数据，并且不会让模型过拟合。我们的实验结果表明，ReLU的性能会略好于tanh，而tanh又明显比sigmod要好。  

![Figure3](/pictures/1TranslateNeuralCollaborativeFiltering_Figure3.png) 

至于网络结构的设计，一个常见的解决方案是按照塔式样式（tower pattern），其中底层是最宽的，随后的依次每一层的神经元个数都会减少（如图2所示）。以更高层使用少量隐藏单元为基础，它们能够学习到更多数据的抽象特征。我们主要按照经验实现了塔式结构，从低到高，逐层神经元数量减半（halving the layer size for each successive higher layer）。

### 3.4 GMF和MLP的混合

到目前为止，我们使用线性内核（linear kernel）对隐性特征之间的交互进行建模，得到了NCF和GMF两个实例，因此MLP使用非线性内核（a non-linear kernel）从数据中学习得到交互函数（interaction function）。问题随之出现：在NCF框架下如何融合GMF和MLP，他们能够互相增强彼此，从而更好的模拟复杂user-item交互。

一个直接的方案是让GMP和MLP共享相同的embedding layer，并且组合它们的交互函数输出。这个方法和著名的神经向量网（Neural Tensor Network, NTN）有相同的思想。具体而言，GMF和单层MLP组合的模型可以形式化的定义如下：  
$$\hat{y}_{ui}=\sigma (h^T a(p_u \cdot q_i + W \begin{bmatrix}
    p_u \\
    q_i \\
\end{bmatrix}  +b)) \tag{11}$$
尽管共享GMF和MLP的embeddings可能会限制混合模型的性能。例如：它意味着GMF和MLP必须使用长度相同的embeddings；对于数据集而言两个模型的最优embedding大小的差别是非常大的（varies a lot），这个解决方法可能无法获得最优的集成模型。

|编号|英语|中文|理解|
|---|---|---|---|
|1|w.r.t.|with respect to 的缩写。是 关于；谈及，谈到的意思|[参考](https://blog.csdn.net/qq_28193019/article/details/88087158)|
|2|hyper-parameter|超参数|可以理解为完全用于调整模型状态和值的一些人工调试参数。这些参数不是从数据中或者模型中来的，而是由人工设置而来的。|

为了给混合模型提供更多的灵活性，我们允许GMF和MLP学习各自的embedding，并且通过两个模型的隐藏层将它们组合起来。图3详细说明吗了我们的计划，形式化的定义如下：  
$$
\phi^{GMF}=p_u^G \cdot q_i^G, \\
\phi^{MLP}=a_L(W_L^T(a_{L-1}(...a_2(W_2^T \begin{bmatrix}
    p_u^M \\
    q_i^M \\
\end{bmatrix} + b_2)...))+b_L), \\
\hat{y}_{ui} = \sigma (h^T \begin{bmatrix}
    \phi^{GMF} \\
    \phi^{MLP}
\end{bmatrix}), \tag{12}
$$  
其中$p_u^G$和$p_u^M$分别定义了GMF和MLP的user embedding部分；与之类似的$q_i^G$和$q_i^M$分别定义了item embeddings。如前所述，我们使用ReLU作为MLP层的激活函数。这个模型组合了MF的线性和DNNs的非线性来模拟user-item隐性结构。我们将这个模型命名为Neural Matrix Factorization，缩写为"NeuMF"。模型中关于每个节点的参数都可以通过标准反向传播来计算，由于文章体量限制，我们这里省略了具体说明。

#### 3.4.1 预训练

由于NeuMF目标函数的非凸性，基于梯度的优化方法只能找到局部最优解。按文献中的说明，初始值对深度学习模型的收敛和性能起着决定性的作用。由于NeuMF是GMF和MLP的组合，我们建议使用GMF和MLP的预训练模型初始化NeuMF。

我们首先用随机初始值训练GMF和MLP直至收敛。然后，我们使用它们的模型参数作为NeuMF参数相应部分的初始化。唯一的调整是在输出层上，我们将两个模型的权重调整为：
$$h\leftarrow \begin{bmatrix}
    \alpha h^{GMF} \\
    (1-\alpha)h^{MLP} \\
\end{bmatrix} \tag{13}$$  
其中$h^{GMF}$和$h^{MLP}$分别表示为GMF和MLP的预训练$h$向量；$\alpha$是一个超参数用于平衡两个预训练模型。

|编号|英语|中文|理解|
|---|---|---|---|
|1|from scratch|从头开始，从零开始|/|
|2|Adaptive Moment Estimation|自适应矩估计|还不理解，[参考](https://www.zhihu.com/question/323747423/answer/790457991)|
|3|vanilla SGD|朴素SGD|[参考](https://blog.csdn.net/zxrttcsdn/article/details/79994730)|
|4|outperform|优于|/|
|5|update on/in|用on时后面接具体到天，in后面接月、季度、年。|[参考](http://www.360doc.com/content/18/1202/16/43864282_798777797.shtml)|

为了从头开始训练GMF和MLP，我们采用了Adaptive Moment Estimation(Adam)，它通过对频繁使用的参数进行比较小的更新，同时对不频繁使用的参数进行比较大的更新。Adam对于两个模型来收敛速度都要比朴素SGD要快，并且减少了调整学习率的代价。将预训练好的参数输入NeuMF后，我们使用朴素SGD而不是Adam对其进行优化。这是因为Adam需要保存动量信息以正确更新参数。由于我们仅使用预先训练的模型参数初始化NeuMF，并且放弃保存动量信息，因此不适合使用基于动量的方法进一步优化NeuMF。（这段的内容就一个意思：使用adam进行预训练，使用vanilla SGD进行优化（也就是在数据集上训练））。

## 4. 实验

在本节中，我们实验的目标是回答一下几个研究问题：
RQ1 我们提出的NCF方法是否优于目前最先进的隐性协同过滤方法？
RQ2 我们提出的优化框架（基于负样本的log loss）如何应用于推荐任务中？
RQ3 更深层的隐藏单元（deeper layers of hidden units）是否有助于从user-item交互数据中学习？
在下面的内容中，我们首先介绍了实验设置，然后回答了上述3个研究问题。

### 4.1 实验设置

|编号|数据集名称|链接|
|---|---|---|
|1|MovieLens|[链接](http://grouplens.org/datasets/movielens/1m/)|
|2|Pinterest|[链接](https://sites.google.com/site/xueatalphabeta/academic-projects)|

|编号|英语|中文|理解|
|---|---|---|---|
|1|Sparsity|稀疏|表格中用这个还没有理解。|
|2|evaluating content-based image recommendation|基于内容图像的推荐系统|也就是通过item的图像来做推荐系统。这里需要和pinterest的功能联系起来才能理解。这里面的图片就是用户需要的商品。|
|3|pin|钉到板子上的动作|pinterest是一个图片社交分享网站，用户将感兴趣的图片放到自己的面板上称其为pin|
|4|leave-one-out evaluation|留一法|这种方法比较简单易懂，就是把一个大的数据集分为k个小数据集，其中k-1个作为训练集，剩下的一个作为测试集，然后选择下一个作为测试集，剩下的k-1个作为训练集，以此类推。这其中，k的取值就比较重要，在书中提到一般取10作为k的值（具体原因 不太清楚）。这种方法也被叫做‘k折交叉验证法（k-fold cross validation）’。最终的结果是这10次验证的均值。此外，还有一种交叉验证方法就是留一法（Leave-One-Out，简称LOO），顾名思义，就是使k等于数据集中数据的个数，每次只使用一个作为测试集，剩下的全部作为训练集，这种方法得出的结果与训练整个测试集的期望值最为接近，但是成本过于庞大。[参考](https://blog.csdn.net/weixin_35436966/article/details/98494046)|
|5|Normalized Discounted Cumulative Gain, NDCG|归一化折扣累积增益|是用来衡量排序质量的指标[参考](https://blog.csdn.net/xiangyong58/article/details/51166127)。**具体方法还需要搞清楚**。|
|6|Hit Ratio, HR|命中率|/|
|7|metric|指标|/|

**数据集**. 我们的实验使用两个易于访问的公开数据集：MovieLens和Pinterest。两个数据集的特征如表1所述。

|Dataset|Interaction#|Item#|User#|Sparsity|
|---|---|---|---|---|
|MovieLens|1,000,209|3,706|6,040|95.53%|
|Pinterest|1,500,809|9,916|55,187|99.73%|

1. MovieLens. 这是一个电源评论数据集，它被广泛用于评估协同过滤算法。我们用的版本包含约一百万条评论，其中每个用户至少有20条评论。显然这是一个明确的反馈数据，但是我们有意从明确的反馈信息当中的隐性信号来研究学习的性能（也就是将显性反馈故意转化为隐性信号来进行研究）。最终，我们通过将每个条目用户是否对其评论将其标识为0或者1来将显性反馈转换为隐性数据。
2. Pinterest. 这个数据集是隐性反馈数据。它是评估基于内容的图像推荐系统数据由组成的。

原始数据非常大但是非常稀疏。例如：有超过20%的用户只有一个pin，这导致能来评估协同过滤算法。因此，我们以和MovieLens相同的方法来过滤数据集，这种方法是只保留至少有20个交互的（pins）的用户。这导致数据的子集只包含了55,185个user和1,500,809条交互（评论）。每个交互意味着用户是否已经将图片放到了自己的面板上。

**评估协议（Evaluation Protocols）** 为评估item推荐系统的性能，我们使用了文献中广泛使用的leave-one-out evaluation。对于每个用户而言，我们将其最新的交互作为测试集，剩余的数据作为训练集。由于在评估过程中为每个user的所有items都进行排序过于耗时（time-consuming），所以我们使用通用策略来优化这个过程，这个策略是：随机抽取没有和user有交互的item，在100个items中对测试item进行排名（这个地方的策略还没理解）。排名list的性能由命中率（Hit Ratio, HR）和Normalized Discounted Cumulative Gain来判断。在没有特别说明得情况下，我们将这两个指标的排名列表都设置为10行。因此，命中率可以直观的衡量测试item是否出现在前10名中，**NDCG通过分配更高的分值给点击排名前几个的情况来说明点击的位置**。我们计算了每个测试user的两个指标，并且以平局值作为结果（reported the average score）。（这一段主要是表明模型的评价方法）

|编号|英语|中文|理解|
|---|---|---|---|
|1|Baselines|基线|就是参照物的意思。[参考](https://www.zhihu.com/question/313705075)|
|2|be worth note|值得注意的是|/|
|3|without special mention|不需要特别注意|/|
|4|predictive factor|预测因子|输出向量的维度|
|5|evaluated the factors|评估因子|/|

**参照物Baselines** 我们将通过以下几种方法来比较我们提出的NCF方法（包括GMF、MLP和NeuMF）：

- ItemPop：item通过他们的流行程度来进行排名。流行程度通过交互数据决定。这是一个公开的方法，用于对推荐的性能进行基准测试。
- ItemKNN：这是一个标准的基于物品的协同过滤方法。我们按照参考文献19的设置对隐性数据进行调整。
- BPR：这个方法优化了公式(2)所示的MF模型。并且该方法使用了pairwise ranking loss，该loss专门用于从隐性反馈中学习。这是一个具有非常高竞争力的item推荐基准。我们使用固定的学习率，通过改变学习来对性能进行比较，从中选择性能最好的。
- eALS：这是一种最先进的item推荐MF方法。它优化了等式(5)的平方损失，将所有未观察到的交互视为负面实例，并且按照item流行度对其进行非均匀的加权。由于eALS的性能要由于均匀加权法WMF，所以我们不会再说明WMF的性能了。

由于我们提出的方法的目标是对user和item之间的关系进行建模，因此我们主要和user-item类模型进行比较。我们省略了与item-item类模型（SLIM和CDAE）的比较，因为性能差异可能是由于定制化的user模型导致的，所以和这类模型不存在比较的基础。

**参数设置**：[我们基于Keras实现了我们提出的方法](https://github.com/hexiangnan/neural_collaborative_filtering)。为了确定NCF方法的超参数，我们随机对每个用户的所有交互数据抽样一条作为验证数据，并在验证数据上调整超参数。所有NCF模型都是通过优化公式(7)所示的log loss进行学习的。在公式(7)中，我们对每个正实例取4个的负实例（这里用的是instance这个单词，我理解就是样本）。对于从头开始训练的NCF模型，我们使用高斯分布来随机初始化模型的超参数（其中平均值为0，标准差为0.01），使用mini-batch Adam来优化模型。我们测试的batch大小为[128, 256, 512, 1024]并且学习率为[0.0001, 0.0005, 0.001, 0.005]。由于NCF的最后一个隐藏层决定了模型的能力，我们将其称为预测因子（predictive factors），我们评估了值为[8, 16, 32, 64]factor的性能（~~这里要么评估的是隐藏层的层数，要么评估的是隐藏层中节点个数。结合后面的内容，这里应该是隐藏层中节点的数量。~~ ~~前面两种可能都不对，这个factor好像类似权重的意思，以为后面说了设置factor为一个数。好像也不对，后面用的是size of predictvie factors来描述它。这个需要看代码来明确。~~ 输出层输出的结果向量的维度）。值得注意的是，较重要factor可能导致过拟合并且降低性能。另外，我们为MLP配置了3个隐藏层；例如，如果predictive factors的大小为8，那么neural CF层数量就是$32\rightarrow 16\rightarrow 8$，嵌入层大小是16。对于使用预训练的NeuMF，$\alpha$设置为0.5，允许预训练GMF和MLP对NeuMF的初始化做出相同的贡献。

### 4.2 性能比较(RQ1)

图4显示了不同数量predictive factors对于HR@10和NDCG@10性能的影响。对于BPR和eALS方法而言，predictive factors的数量等于latent factors的数量。对于ItemKNN而言，我们测试了不同的邻居大小对性能的影响，同时说明了额最佳性能时的选择。由于ItemPop的性能比较弱，为了更好的突出设计方法的性能差异，我们有意忽略了它。所以在图4中没有显示。

![Performance of HR@10 and NDCG@10](/pictures/1TranslateNeuralCollaborativeFiltering_Figure4.png)

首先，我们可以看到NeuMF在两个数据集上都取得了最好的性能，而且大大优于目前最先进的eALS和BPR方法（平均而言，相对于eALS和BPR的相对改善率为4.5%和4.9%）。对于Pinterest而言，即使使用大小较小（为8）的predictive factor，NeuMF也显著优于eALS和BPR的表现，predictive factor大小较大时（为64）也明显优于eALS和BPR。这表明，作为融合了线性MF和非线性MLP的NeuMF模型具备很高的表达（high expressiveness）能力。

其次，另外两种NCF方法-GMF和MLP也表现出了相当强的性能。其中MLP的表现略逊于GMF。需要注意的是，MLP可以通过添加更多的隐藏层来进一步改进性能（详见4.4节），这里我们只展示了使用个3层次的性能。对于较小的predictive factors，在两个数据集善GMF都由于eALS；虽然GMF会因为较大factors而导致过拟合，但是它获得的最佳性能依然优于或者等于eALS。

最后，由于GMF和BPR学习相同的MF模型，但是使用不同的目标函数，同时不得不说在推荐任务中classificationaware log loss是十分有效的，所以GMF显示出比BPR更具备可持续的改进空间。

![Evaluation of Top-K item recommendation](/pictures/1TranslateNeuralCollaborativeFiltering_Figure5.png)

图5显示了Top-K推荐列表性能，其中排名位置K的范围从1到10。为了使图更清晰，我们只展示了NeuMF的性能，而不是所有三种NCF的方法。与其他方法相比我们可以看出，NeuMF在不同条件（across positions）下都表现出了改善，我们进一步将one-sample和t-tests进行了配对，验证了所有改善在统计学上均具有显著性（$p<0.01$）。就基线方法而言，eALS在MovieLens上表现优于BPR，相对改善率为5.1%，而在NDCG方面则不如Pinterest上的BPR，这与[14]的发现是一致的，既BPR在ranking性能的体现，owing to its pairwise ranking-aware learner。基于领域ItemKNN的性能不如model-based方法。ItemPop的表现最差，这表明有必要对用户的个性化偏好进行建模，而不仅仅是推荐流行的item给用户。

#### 4.2.1 预训练的效用

为了证明NeuMF预训练的实用性，我们比较了有和没有预训练两个版本的NeuMF的性能。对于没有预训练的NeuMF，我们使用Adam通过随机初始化权重来完成学习。如表2所示，具有雨荨的NeuMF在大多数情况下都能获得更好的性能；只有predictive factors 为8对于MovieLens数据集的情况下，预训练方法的性能表现稍差。对于MovieLens和Pinterest两个数据集，NeuMF预训练相比于没有预训练的情况，改善率为2.2%和1.1%。这一结果证明了我们采用预训练方法对初始化NeuMF是有效的。

![Performance of NeuMF with and without pre-training](/pictures/1TranslateNeuralCollaborativeFiltering_Table2.png)

### 4.3 负采样的对数损失（log loss with negative sampling）

|编号|英语|中文|理解|
|---|---|---|---|
|1|one-class nature|单分类的天然属性|由于在隐性推荐中缺少负样本，在只有正样本的情况下，对item进行推荐就是一种单分类场景。但是单分类问题在工业界广泛存在，由于每个企业刻画用户的数据都是有限的，很多二分类问题很难找到负样本，即使用一些排除法筛选出负样本，负样本也会不纯，不能保证负样本中没有正样本。所以在只能定义正样本不能定义负样本的场景中，使用单分类算法更合适。[参考](https://www.cnblogs.com/wj-1314/p/10701708.html)|
|2|on par with|与...平分秋色/势均力敌/不分上下|/|

![Training loss and recommendation performance of NCF methods](/pictures/1TranslateNeuralCollaborativeFiltering_Figure6.png)

为了应对隐式反馈的one-class的天然属性，我们将推荐转换为二分类任务。通过将NCF视为一个概率模型，我们使用Log loss作为优化函数。图6显示了在MovieLens数据集上使用NCF方法每次迭代的训练损失（对所有实例取了平均值）和推荐性能。在Pinterest数据集上的结果显示了基本相同的趋势，由于文章篇幅所限所以我们没有在本文中展示出来。首先，我们可以看到，随着迭代次数的增加，NCF模型的训练损失逐渐减少，推荐性能逐渐提高。效果明显变好的情况发生在前10次迭代中，更多的迭代次数可能会使得模型过拟合（例如：尽管NeuMF的训练损失在前10次迭代之后不断减少，但其推荐性能反而在下降）。其次，在三种NCF方法中，NeuMF的训练损失最小，其次是MLP，最后是GMF。推荐性能也表现出于NeuMF>MLP>GMF的情况。上述研究结果表明：使用log loss作为学习隐性数据的优化函数的合理性和有效性。

![Performance of NCF methods w.r.t. the number of negative samples per positive instance](/pictures/1TranslateNeuralCollaborativeFiltering_Figure7.png)

pointwise log loss相比于pairwise objective functions的一个优势是可以灵活的定义负实例的采样率。虽然pairewise objective functions只能将一个负实例和一个正实例配对，但我们可以灵活的控制pointwise loss的采样率（这句话没有明白逻辑关系的重点在哪里？）。为了说明负采样对NCF方法的影响，我们在图7中展示了不同负采样率对NCF方法性能的影响。可以清楚的看到，每个正实例仅仅采样一个负实例是不足以实现最佳性能的，因此采样更多的负实例是有益处的。GMF和BPR相比，我们可以看到，采样率为1的GMF的性能与BPR相当，而在采样率较大的情况下GMF的性能明显优于BPR。

这显示了pointwise log loss比pairwise BPR loss更具优势。对于两个数据集而言，最佳的采样率约为3到6。在pinterest上，我们发现当采样率大于7时，NCF方法的性能开始下降。它表明过于设置过高的采样率对性能会产生不利影响。

### 4.4 深度学习是否有用？（RQ3）

|编号|英语|中文|理解|
|---|---|---|---|
|1|identity function|恒等函数|$f(x)=x$|

由于将神经网络用于学习use-item交互函数的工作很少，所以很好奇使用深度网络结构是否有利于推荐任务。为此，我们进一步研究了具有不同隐藏层数的MLP。结果总结在表3和表4中。MLP-3展示了具有3个隐藏层的MLP范范（除嵌入层之外），和其他类似的符号。正如我们所见，即使对于具有相同功能的模型，堆叠更多层对性能更有利的。这个结果非常令人鼓舞，表明使用深度模型进行协同推荐的有效性。我们将这种改进归功于堆叠更多非线性层而带来的非线性性。为了验证这一点，我们进一步尝试堆叠线性层，使用identity function作为激活函数，它的性能比使用ReLU作为激活函数差得多（这里是否使用更多的层数来验证这一点合适一些？或者换一些复杂一点的线性函数是否会有变化？）

![HR@10 of MLP with different layers](/pictures/1TranslateNeuralCollaborativeFiltering_Table3.png)

![NDCG@10 of MLP with different layers](/pictures/1TranslateNeuralCollaborativeFiltering_Table4.png)

对于没有隐藏层的的MLP-0（即嵌入层直接投影到预测），性能非常弱，而且并不比non-personalized ItemPop好。这验证我们再第3.3节中所讨论的论点，即简单的连接user和item的latent vectors不足以对其交互特征建模，因此有必要将其转换为隐藏层。

## 5. 相关工作

虽然早期与推荐系统相关的文献主要集中在显式反馈，但最近注意力越来越转向隐性反馈数据了。具有隐性反馈的协同过滤（CF）任务通常被描述为一个item 推荐问题（物品推荐问题），其目的是向用户推荐一个简要的item列表。通过显式反馈数据已经被普遍解决的评分预测（rating prediction）相比，解决item推荐问题是更实际但是更有挑战性的问题（潜台词是通过隐性数据的item推荐问题）。一个关键性的领悟（突然体会到，insight）是对缺失数据如何进行建模，这项工作在显示反馈中常常被忽略了。为了通过隐性反馈对item推荐制作隐性特征模型，早期的工作采用了统一的权重，其中提出了两种策略，其一：将素有缺失数据视为负实例；其二，从缺失数据中抽取负实例。最近He等人和Liang等人提出了加权缺失数据（weight missing data，这里也可以翻译为权重缺失数据）的专用模型。Rendle等人为feature-based factorization models开发了隐式坐标下降(implicit coordinate descent, iCD)的解决方案，达到了item推荐的最优性能。下面，我们将讨论在推荐系统中使用神经网络的情况。

|编号|英语|中文|理解|
|---|---|---|---|
|1|implicit coordinate descent, iCD|隐式坐标下降|[需要了解](https://blog.csdn.net/qq_32742009/article/details/81735274)|
|2|Restricted Boltzmann Machines|限制玻尔兹曼机|[需要了解](https://blog.csdn.net/qq_39388410/article/details/78306190)|
|||||

Salakhutdinov等人早期的先驱性的工作中提出了一种2层Restricted Boltzmann Machines(RBMs)，RBMs被用于模拟用户对物品的明确评级。这项工作后来被推广到模拟评级的顺序属性（ordinal nature of ratings）。最近，自动编码器（autoencoders）已成为构建推荐系统的流行选择。user-based AutoRec的思想是学习隐藏结构，根据用户的历史评分作为输入能够重现一个用户的评价。就用户的个性化而言，