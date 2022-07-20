# Inductive Learning and Transductive Learning

也称其为 Inductive Problem and Transductive Problem。

## 参考

1. [inductive和transductive的理解](https://blog.csdn.net/zfhsfdhdfajhsr/article/details/119059065)
2. [如何理解 inductive learning 与 transductive learning?](https://www.zhihu.com/question/68275921)
3. [图解inductive+transductive ML](https://zhuanlan.zhihu.com/p/455808338)

## statement

都是参考中内容的复制。

## 说明

### 参考2中的说明

Inductive learning，翻译成中文可以叫做“归纳式学习”，顾名思义，就是从已有数据中归纳出模式来，应用于新的数据和任务。我们常用的机器学习模式，就是这样的：根据已有数据，学习分类器，然后应用于新的数据或任务。

Transductive learning，翻译成中文可以叫做“直推式学习”，指的是由当前学习的知识直接推广到给定的数据上。其实相当于是给了一些测试数据的情况下，结合已有的训练数据，看能不能推广到测试数据上。

对应当下流行的学习任务：

Inductive learning对应于meta-learning (元学习)，要求从诸多给定的任务和数据中学习通用的模式，迁移到未知的任务和数据上。

Transductive learning对应于domain adaptation (领域自适应)，给定训练的数据包含了目标域数据，要求训练一个对目标域数据有最小误差的模型。

### 参考1中的说明

最近在阅读论文的过程中，文章中提到了inductive和transductive问题，在此记录一下他们各自的意义。

inductive是归纳的意思，指的是从特殊到一般的学习。Inductive learning 是从特定任务到一般任务的学习，实际上，我们传统的supervised learning都可以理解为是Inductive learning的范畴：基于训练集，我们构建并训练模型，而后将其应用于测试集的预测任务中，训练集与测试集之间是相斥的，即测试集中的任何信息是没有在训练集中出现过的。即模型本身具备一定的通用性和泛化能力。

再看其关于Transductive的定义：Transduction is reasoning from observed, specific (training) cases to specific (test) cases.大家先理解下上面这句话，其中的obeserved其实同时修饰着后面的training cases和test cases。相比Inductive learning，Transductive learning拥有着更广的视角，在模型训练之初，就已经窥得训练集（带标签）和测试集（不带标签），尽管在训练之时我们不知道测试集的真实标签，但可以从其特征分布中学到些额外的信息（如分布聚集性），从而带来模型效果上的增益。但这也就意味着，只要有新的样本进来，模型就得重新训练。

综上，总结一下这二者的区别：模型训练：Transductive learning在训练过程中已经用到测试集数据（不带标签）中的信息，而Inductive learning仅仅只用到训练集中数据的信息。模型预测：Transductive learning只能预测在其训练过程中所用到的样本（Specific --> Specific），而Inductive learning，只要样本特征属于同样的欧拉空间，即可进行预测（Specific --> Gerneral）模型复用性：当有新样本时，Transductive learning需要重新进行训练；Inductive Leaning则不需要。模型计算量：显而易见，Transductive Leaning是需要更大的计算量的，即使其有时候确实能够取得相比Inductive learning更好的效果。其实，我们仅从它们的字面意思上也可以有些理解，Inductive一般翻译做归纳式，归纳是从特殊到一般的过程，即从训练集中学习到某类样本之间的共性，这种共性是普遍适用的。Transductive一般译作直推式，则显得僵硬许多，意味着必须知道它要推论的所有case长什么样时才能work

最后尽管有了这么多解释，能理解透彻的还是inductive问题，因为用的模型接触的问题都是inductive。transductive问题需要在找个模型理解下，继续补充文章。
可以着重理解参考文献3篇。