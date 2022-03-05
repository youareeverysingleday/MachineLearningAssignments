# 机器学习

一些基本概念。只做参考。可以快速的过一遍。
这门课内容是按照周志华《机器学习》来讲的，但是其中很多部分并没有讲。

1. 所有没有理解的地方通过~~XXX~~进行了表示。

## 1. 机器学习概述

1. 概念：机器学习研究的是计算机怎样模拟人类的学习行为，以获得新的知识或技能，并重新组织已有的知识结构使之不断改善自身。就是计算机从数据中学习出规律和模式，以应用在新数据上做预测的任务。
2. 能解决的问题：聚类、分类、回归、强化学习。（实际上就是两类：分类和回归，分别对应离散值和连续值）。聚类是无监督学习，挖掘数据的关联关系。强化学习主要用于互动环境中。
3. 机器学习分类：
   1. 监督学习，特征和标签。
   2. 无监督学习，关联规则。
   3. 强化学习，从环境到行为映射的学习。
4. 工作阶段：数据预处理（特征抽取，幅度缩放，特征选择，降维，采样），模型学习（模型选择，交叉验证，结果评估，超参选择，模型训练），模型评估，新样本预测（机器学习实用阶段）。
5. 评估指标：错误率低，准确率高。
6. 数据采样的方法：
   1. 留出法：保持数据分布一致，多次重复划分，测试集不能太大和太小。
   2. k折交叉验证。
   3. 自助采样法。对样本进行有放回重复采样，取到的数据作为训练集，没有取到的数据作为测试集。
7. 度量标准：
   1. 性能度量：衡量模型泛化能力的数值评价标准。回归问题常采用的是均方误差。
   2. 分类问题常用的性能度量：错误率和精度。
      1. 错误率：分类错误的样本数除以总数。
      2. 精度：分类准确的样本数除以总数。
      3. 混淆矩阵：二分类混淆矩阵。查准率和查全率。
      4. $F_1$值和$F_{\beta}$。
      5. ROC和AUC，曲线是ROC，曲线下的面积就是AUC。
      6. 回归问题的度量标准：
         1. 平均绝对误差(Mean Absolute Error)$\text{MAE}=\frac{1}{n} \sum\limits_{i=1}^{n}|f_i -y_i|$。
         2. 均方误差(Mean Square Error)$\text{MSE}=\frac{1}{n} \sum\limits_{i=1}^{n}(f_i -y_i)^2$。
         3. 方根误差(Root Mean Square Error)$\text{RMSE}=\sqrt{\text{MSE}}$。
         4. R平方$r^2=1-\frac{SS_{res}}{SS_{tot}}=1-\frac{\sum(y_i-f_i)^2}{\sum(y_i-\overline{y})^2}$。
   3. 机器学习的目标：找到具有泛化能力的“好模型”（这句话不准确，泛化能力强的情况下，大概率其他性能就会下降，实际上是对应不同任务性能就会有所倾斜。应该是综合能力好的模型）。
8. 机器学习算法一览：
   1. 非监督算法：
      1. 对于连续值（continuous）：聚类和降维算法（Clustering and Dimensionality Reduction）。具体包含：SVD，PCA，K-Means。
      2. 对于离散值（分类值，categorical）。
         1. 关联分析（association analysis），具体包含Apriori和FP-Growth。
         2. 马尔科夫链（Hidden markov model）。
   2. 监督算法：
      1. 对于连续值：回归（）决策树，随机森林。
      2. 对于离散值（分类值，categorical）。
         1. 分类具体包含：KNN，Trees，Logistic Regression，Naive-Bayes，SVM。
9. 一般选择模型的流程：
   |![GeneralProcessofModelSelection](../../pictures/GeneralProcessofModelSelection.jpg "一般选择模型的流程")|
   |:--:|
   | *1.1* |

   ```mermaid
    graph TD
        A[start]-->B{数据量是否少于50}
        B-->|no|C[补充数据]
        B-->|yes|D{分类还是回归问题要求输出的是连续值还是离散值}
        D-->|分类|E{数据中是否含有标签数据}
        D-->|分类|F{数据中是否含有标签数据}
        E-->|有标签监督|G[classification分类问题]
        E-->|没有标签无监督|H[clustering聚类问题]
        F-->|有标签|I[regression回归问题]
        F-->|没有标签|J[dimenisonality reduction降维问题]
    ```

10. 不同的算法对相同的问题有不同的处理方法。不同的算法带来的决策边界是不一样的。 回归问题有不同的拟合方式。

## 2. 线性回归和逻辑回归

1. 线性模型（linear model）。特点：简单、基本、可解释性好。通过样本属性的线性组合来进行。
   1. 分类：通过一条线来将两类数据分开。
   2. 回归，通过一条线对所有数据进行拟合。
2. 损失函数loss function。
3. 通过损失函数就将回归问题转换为了优化问题。即为对凸函数求极值。
4. 梯度下降法来求凸函数的极值。梯度是决定迭代的方向。迭代的步长通过其他方法来决定。对于一元函数的损失函数计算方法$\theta_1 = \theta_1-\alpha \frac{dJ(\theta_1)}{d\theta_1}$，其中步长通过$\alpha$决定，方向由$\frac{dJ(\theta_1)}{d\theta_1}$决定。
   1. 梯度就是损失函数的切线方向。
   2. 超参数$\alpha$决定步长。
   3. 每次都需要更新$\theta$，最终找到最优$\theta$。
5. $\alpha$也称为学习率，不能太小也不能太大。
6. 欠拟合和过拟合。欠拟合好解决，过拟合不好解决。
   1. 欠拟合：模型没有很好的捕捉到数据特征，不能很好的拟合数据。
   2. 过拟合：把样本中的一些噪声特性也学习了下来，泛化能力差。
7. 减小过拟合的方法：正则化。通过正则化添加参数“惩罚”，控制参数幅度限制参数搜索空间，减小过拟合风险。原始的损失函数是：$J(\theta)=\frac{1}{2m} \sum \limits_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})^2$修改为了$J(\theta)=\frac{1}{2m} [\sum \limits_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})^2 + \lambda \sum \limits_{i=1}^{m}\theta_i^2]$。将模型的参数添加到了损失函数中，期望模型的参数越小越好。
   1. 一般而言是参数越少那
   2. 么模型的复杂度就越低，通过将参数设置为0就可以降低模型的复杂度。
   3. 从数学的角度而言：
   4. 为什么$l_1$正则化具有稀疏性？为什么$l_1$可以进行特征选择？[详细说明](../ReadPapers/8Regularization.md)
   5. $l_1$和$l_2$之间的区别？
8. 广义线性模型，比如$lny=\boldsymbol{w}^T \boldsymbol{x}+b$，这样就得到了对数线性回归。
9. 逻辑回归：
   1. 线性回归与逻辑回归的关系。在逻辑回归中并不会去拟合样本的分布，而是确定决策边界。包括线性决策边界和非线性决策边界。这里需要说到的是sigmod函数作为非线性函数来对线性值进行映射。
      1. 线性的对应的就是的直线函数。
      2. 非线性的对应的就是各种非直线的函数，比如圆形、抛物线等等。
   2. 逻辑回归的决策边界。
   3. 逻辑回归损失函数。
      1. 损失函数不能再使用均方差损失（MSE），这样可能导致出现局部最优解（local cost minimum）。而实际上是期望求得全局最优解（global cost minimum）的。因为在局部最优解的时候梯度已经为0了（也就是说对应的损失函数不是凸函数）。
      2. 使用的损失函数是对数损失/二元交叉熵损失（~~最大似然到对数损失，这个位置不清楚~~）
         $$
         Cost(h_{\theta}(x),y) = \left\{
         \begin{aligned}
         & -log(h_{\theta}(x)) \text{, if y=1}\\
         & -log(1-h_{\theta}(x)) \text{, if y=0}
         \end{aligned}
         \right.
         $$
      3. 损失函数与正则化。依旧存在过拟合问题，决策边界可能抖动得很厉害。(下面的公式可能有问题。)
         $$
         \begin{aligned}
         &\text{损失函数：} \\
         &J(\theta)=\frac{1}{m} \sum\limits_{i=1}^{m}Cost(h_{\theta}(x_i),y_i)\\
         &=-\frac{1}{m} [\sum\limits_{i=1}^{m}y_i logh_{\theta}(x_i)+(1-y_i)log(1-h_{\theta}(x_i))]\\
         &\text{添加正则化项之后的损失函数：}\\
         &J(\theta)=-\frac{1}{m} [\sum\limits_{i=1}^{m}y_i logh_{\theta}(x_i)+(1-y_i)log(1-h_{\theta}(x_i))] + \frac{\lambda}{2m}\sum\limits_{j=1}^{m}\theta_j^2\\
         & \text{使用梯度下降的方法来求最小值：}\\
         & \theta_j=\theta_j-\alpha\frac{\partial J(\theta)}{\partial \theta_j}
         \end{aligned}\\
         $$
         同样是使用梯度下降的方法来求最小值。
   4. 从二分类到多分类。有两种思路，这两种思路共同特点就是将多分类问题转换为二分类问题来解决。
      1. 思路1：将每个类别和除该类别之外的认为是两类，然后分类。针对所有类别逐一做二分类。
      2. 思路2：~~显然没有说清楚（理解好像是对每两个类别之间做一个分类器，也就是两两分类）~~。
10. 工程应用经验
    1. 逻辑回归和其他模型。
       1. 逻辑回归的特点：
          1. LR能以概率的形式输出结果，而非0，1判定。
          2. LR的可解释性强，可控度高。
          3. 训练快，特征工程（feature engineering）之后效果很好。
          4. 因为结果是概率，可以做排序模型。
          5. 添加特征非常简单。
       2. 应用
          1. CTR预估和推荐系统的learning to rank各种分类场景。
          2. 很多搜索引擎厂的公告CTR预估基线版是LR。
          3. 电商搜索排序/广告CTR预估基线版是LR。
          4. 新闻APP的推荐和排序基线也是LR。
    2. 样本处理。
       1. 样本特征处理：离散化后用独热向量编码（one-hot encoding）处理成0，1值。LR训练连续值，注意做幅度缩放（scaling，不同特征的取值需要在同一个范围之内）。
       2. 处理大样本量：事实spark或者MLib，试试采样（注意是否需要分层采样）。
       3. 注意样本平衡：对样本分布敏感（不能使得不同类别的样本的分布不均匀）。通过欠采样和过采样来处理不同的样本数量。另外也可以修改损失函数给不同的样本以不同的权重来解决样本不平衡问题。
    3. 工具包和库。
       1. 常用python库sklearn。
       2. python绘图库

         ```Python
         from mpl_toolkits.mplot3d import axes3d
         from sklean.preprocessing import PolynomialFeatures
         poly = PloyomialFeatures(6) #引入多项式特征，用于将二维数据映射到高维空间。
         ```

    4. 正则化系数太大或者太小会出现的情况，分别对决策边界产生的影响。**正则化牺牲了模型的精度提高了泛化性能**。
       1. lambda=0 就是没有正则化，这样的话就会过拟合。
       2. lambda=1 这是正常值。
       3. lambda=100 正则化项太激进，导致基本没有拟合出决策边界。

## 3. 决策树模型概述

1. 决策树模型概述。
   1. 决策树Decision Tree model是一个模拟人类决策过程思想的模型。通过多个条件逐一筛选的对象的过程模拟。
   2. 模型特点：简单、逻辑清晰、可解释性非常好。
   3. 决策树基于树结构进行决策。
      1. 每个内部节点对应于某个属性上的测试。
      2. 每个分支对应于该测试的一种可能结果（即该属性的某个取值）。
      3. 每个叶节点对应于一个预测结果。
   4. 学习过程。通过对训练两边额分析来确定“划分属性”（即内部节点所对应的属性。）
   5. 预测过程：将测试示例从根节点开始，沿着划分属性所构成的“判定测试序列”下行，知道叶节点。
   6. 主流算法：CLS、ID3、C4.5、CART、RF(随机森林，最强到的决策树算法)。
2. 决策树模型分类。
3. 算法流程与最优属性选择方法。
   1. 决策树基本流程。
      1. 总体流程成为“分而治之（divide and conquer）”。
      2. 自根至叶的递归过程。
      3. 在每个中间节点寻找一个“划分”（split or test）属性。
      4. 三种停止条件。
         1. 当前节点包含的样本全属于同一类别，无需划分。
         2. 当前属性集为空，或是所有样本在所有属性上取值相同，无法划分。
         3. 当前节点包含的样本集合为空，不能划分。
   2. 最佳属性选择方法。**决策树算法中最核心的一点在于如何选择最佳属性**。
      1. 通过信息增益和信息增益率来选在最佳属性。
         1. **通过信息增益的大小来选择根节点。这个地方需要看周志华《机器学习》4.2.1中的例子来详细理解**。
         2. 定义决策树的过程就是：**首先寻找最佳属性进行划分；第二然后在划分好的集合中继续寻找最佳属性进行划分，直到能把所有的样本能够区分出来**。
         3. 信息增益存在的问题：对可取值数目较多的属性有所偏好。这个时候采用信息增益率作为信息增益的替代。
         4. 启发式（选在属性的方法）：**先从候选划分属性中找出信息增益高于平均水平的，再从中选取增益率最高的**。
      2. 通过基尼指数（gini index）来选择最佳属性，常用语CART中。
         1. 基尼指数定义：$Gini(D)=\sum\limits_{k=1}^{|y|}\sum\limits_{k'\neq k}p_kp_{k'}=1-\sum\limits_{k=1}^{|y|}p_k^2$。
         2. 反映的是从D中随机抽取两个样例，其类别标记不一致的概率。Gini(D)越小，数据集D的确定性（注意这里是确定性）越高。属性a的基尼指数：$Gini\_index(D,a)=\sum\limits_{v=1}^{|V|}\frac{|D_v|}{D}Gini(D_v)$
         3. **在选择属性集合中，选取那个是划分后基尼指数最小的属性**。
   3. 熵与信息论视角。
      1. 信息熵(information entropy，~~这个部分还需要加强理解~~)。$Ent(D)=-\sum\limits_{k=1}^{|y|}p_k lnp_k$
      2. 信息增益(information gain)。$Gain(D,a) = Ent(D)-\sum\limits_{v=1}^{V}\frac{|D_v|}{D}Ent(|D_v|)$。其中$Ent(D)$是划分前的增益，$\sum\limits_{v=1}^{V}\frac{|D_v|}{D}Ent(|D_v|)$是划分后的增益。$\frac{|D_v|}{D}$第v个分支的权重，样本越多越重要。
      3. 信息增益率。$Gain\_ratio(D,a)=\frac{Gain(D,a)}{IV(a)},\text{ 其中}IV(a)=-\sum\limits_{v=1}^{V}\frac{|D_v|}{D}ln\frac{|D_v|}{D}$，属性a的可能取值数目越多（即V越大），则IV(a)的值通常就越大。
      4. 从信息论的视角和机器学习的视角对（XX）进行对比:
         |信息论的视角|机器学习的视角|
         |---|---|
         |接受信号|特征|
         |信源|标签|
         |平均互信息|特征有效性分析|
         |最大熵模型|极大似然估计|
         |交叉熵|逻辑回归损失函数|
4. 剪枝与控制过拟合。
   1. 剪枝和过拟合操作。
      1. 定义：为了尽可能的正确分类训练样本，有可能造成分支过多，造成过拟合。**剪枝：通过主动去掉一些分支来降低过拟合的风险**。
      2. 基本策略：预剪枝(pre-pruning)：提前终止某些分支的生长。后剪枝（post-pruning）:生成一颗完整树之后，再回头来剪枝。
      3. 剪枝的基本原则：剪枝过程中需要评估剪枝前后决策树的优劣。一般使用留出法来进行评估。
      4. 操作步骤：比如采用后剪枝。
         1. 首先通过训练集生成完整的决策树。
         2. 通过验证集对每个节点之后的精度进行计算。
         3. 从底部开始向顶部进行剪枝。比较每个节点剪枝前后的精度，如果节点剪枝前后的精度下降，那么剪枝；如果升高了或者保持不变那么保留节点。
   2. 预剪枝过程与示例。
      1. 在决策树生成的过程中，基于信息增益准则，在划分节点时，若该节点的划分没有提高其在验证集上的准确率，则不进行划分。
   3. 后剪枝与示例。
   4. 预剪枝和后剪枝的对比：
      1. 时间开销：预剪枝训练时间开销降低，测试时间开销降低。后剪枝训练时间开销增加，测试时间开销降低。
      2. 过拟合风险：预剪枝过拟合凤冈县降低，欠拟合风险增加。后剪枝过拟合风险降低，欠拟合风险基本保持不变。
      3. 泛化性能：后剪枝通常由于预剪枝。
5. 数据案例讲解。

   ```python
   import pandas as pd
   # import数据预处理和特征工程的库。
   import sklearn import preprocessing
   # import决策树的库。
   from sklearn import tree

   # ......

   pandas.get_dummies(feature)
   # 用于将数据特征中的不同类型转换为数值型的类型。
   # 比如将白色人种、黄色人种、黑色人种、棕色人种，通过上面的函数转化为0,1,2,3这种形式。
   clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4)
   # 这个函数有两个参数，第一个参数决定用什么来做分类属性；第二个参数设置决策树的最大深度是多少。
   clf = clf.fit(features.values, label.values)

   # ......

   import pydotplus # 用于查看决策树的结构。
   from IPython.display import display, Image
   # ......
   # 说明这里是做了一个对人的收入是否超过5万元的预测。所以在class_names=['<=50K', '>50K']这个参数的时候填写的类别的名称就是这个。
   dot_data = tree.export_graphviz(clf,
                                    out_file=None,
                                    feature_names=features.colums,
                                    class_names=['<=50K', '>50K'],
                                    filled=True,
                                    rounded=True)
   graph = pydotplus.graph_from_dot_data(dot_data) #用来查看决策树的结构
   display(Image(graph.create_png()))
   ```

## 4. 分类回归数和随机森林

1. 连续值和缺省值的处理
   1. 连续值的处理（这里是对于树模型而言的）。**核心思想是连续属性离散化**。常见的做法是二分法(bi-partition)。也就是在连续值之中找一个点，将其划分为2个部分。每个部分对应一种类型。（~~原话是：n个属性值可以形成n-1个候选划分，当做离散值来处理。这个确实没有理解~~）
   2. 缺失值（missing）的处理。方法之一：仅使用无缺失值来判断样例的优劣。（~~这里出现了一种情况：某些特征上的样本会**同时**出现在多个分支上，而且按照例子进入多个分支上的样本的权重之和还为1。这个地方课程没有说清楚。~~）
   3. 从树到规则的建立。一棵决策树对应于一个“规则集”，每个从根节点到叶节点的分支路径对应于一条规则。可以理解为多个if-else语句的合集。所以决策树的可解释性非常好。而且可以进一步提高泛化能力。
2. 回归树模型及构建方法
   1. 之前说的树结构都是用来做分类的。树结构也可以做回归。**回归树本质上是对空间的划分。也就是将特征空间切分成了不相交的子区域，每个区域预估成该区域样本的均值**。
   2. 回归树和决策树操作步骤上是类似的。不同的地方在于使用RSS来代替了信息熵。$RSS=\sum\limits_{j=1}^{J}\sum\limits_{i\in R_j}(y_i-\widetilde{y}_{R_j})^2$。RSS的计算方法是自顶向下的贪婪式的递归方案。RSS最小化和探索的过程计算量非常巨大。一般采用探索式的递归二分来尝试解决这个问题。
   3. 可以通过正则化项来进行过拟合控制。
3. bagging和随机森林
   1. bootstraping。bootstraping来自于成语“pull up by your own bootstrap”，意思是依靠你自己的资源，称为自助法。它是一种有放回的抽样方法。它是非参数同济中一种重要的估计统计量方差进而进行区间估计的统计方法。bootstrap是现代统计学较为流行的一种统计方法，在小样本时效果很好。通过方差的估计可以构造置信区间等，其运用范围得到进一步延伸。其核心思想和基本步骤如下：
      1. 采用重抽样技术从原始样本中抽取一定数量（自己给定）的样本，此过程允许重复抽样。
      2. 根据抽出的样本计算给定的统计量T。
      3. 重复上述N次（一般大于1000），得到N个统计量T。
      4. 计算上述N个统计量T的样本方差，得到统计量的方差。
   2. bagging。bagging是bootstrap aggregating的缩写。使用了bootstraping的思想。bagging降低了过拟合风险，提高了泛化能力。

      ```mermaid
      graph LR
         A[m个样本训练集]-->B[m个样本采样集1]
         A-->C[m个样本采样集2]
         A-.->SL1[......]
         A-->D[m个样本采样集T]
         B-->E[学习器1]
         C-->F[学习器2]
         SL1-.->SL2[......]
         D-->G[学习器T]
         E-->H[集成学习器]
         F-->H
         SL2-.->H
         G-->H
      ```

      输入样本集$D=\{(x_1,y_1),(x_2,y_2),\cdots,(x_m,y_m)\}$，步骤如下：
         1. 对于$t=1,2,\cdots,T$:
            1. 对训练集进行第t次随机采样，共采集m次，得到包含m个样本的采样集$D_m$。
            2. 用采样集$D_m$训练第t个基学习器$G_t(x)$
         2. 分类场景，则T个学习器投出最多票数的类别为最终类别。回归场景，T个学习器得到的回归结果进行算术平均得到的值为最终的模型输出。
   3. 随机森林（Random Forest）。是一种基于树模型的bagging的优化版本。核心思想依旧是bagging，但是做了一些独特的改进。RF使用CART决策树作为基学习器。对于样本集$D=\{(x_1,y_1),(x_2,y_2),\cdots,(x_m,y_m)\}$，具体的过程如下：
      1. 对于$t=1,2,\cdots,T$:
         1. 对训练集进行第t次随机采样，共采集m次，得到包含m个样本的采样集$D_m$。
         2. 用采样集$D_m$训练第m个决策树模型$G_m(x)$，**在训练决策树模型的节点的时候，在节点上所有的样本特征中选择一部分样本特征，在这些随机选择的部分样本特征中选择一个最优的特征来做决策树的左右子树划分**。
      2. 分类场景，则T个基模型（决策树）投出最多票数的类别为最终类别。回归场景，T个基模型（回归树）得到的回归结果进行算术平均得到的值为最终的模型输出。
4. 数据案例讲解
   1. 第一个示例

      ```python
      # 这个例子的代码是完整的。
      import pandas as pd
      import sklearn import preprocessing
      import sklearn.ensemble import RandomForestRegressor
      from sklearn.datasets import load_boston

      boston_house = load_boston()

      boston_feature_name = boston_house.feature_name
      # 房屋的属性。
      boston_feature = boston_house.data
      # 房屋的价格。
      boston_target = boston_house.target

      # 显示数据集的相关信息
      print(boston_house.DESCR)
      
      help(RandomForestRegressor)
      # n_estimators表示本RF有几棵树，也就是树的数量。输入整数类型。可选参数。default = 10。。
      # criterion本RF的优化目标是什么？也就是选择什么样的损失函数。输入字符串类型。可选参数。default = "mse"。

      rgs = RandomForestRegressor(n_estimators=15)
      rgs = rgs.fit(boston_features, boston_target)

      # 进行预测。
      rgs.predict(boston_features)
      ```

   2. 第二个示例，对连续值特征的数据进行分类（基于决策树）。对鸢（yuan1）尾花进行分类。通过鸢尾花的4个连续值属性来预测花属于哪一类。花的类别一共有3类。

      ```python
      import pandas as pd
      import sklearn import preprocessing
      import sklearn.ensemble import tree
      from sklearn.datasets import load_iris

      iris = load_iris()

      iris_feature_name = iris.feature_names
      iris_features = iris.data
      iris_target_name = iris_target_names
      iris_target = iris.target

      # 构建决策树分类器。
      clf = tree.DecisionTreeClassifier(max_depth=4)
      clf = clf.fit(iris_features, iris_target)

      import pydotplus # 用于查看决策树的结构。
      from IPython.display import display, Image

      dot_data = tree.export_graphviz(clf,
                                    out_file=None,
                                    feature_names=iris_feature_name,
                                    class_names=iris_target_name,
                                    filled=True,
                                    rounded=True)
      graph = pydotplus.graph_from_dot_data(dot_data) #用来查看决策树的结构
      display(Image(graph.create_png()))

      ```

## 5. 支持向量机

一般而言，机器学习算法分类有以下几个指标：

1. 分类问题/回归问题。
2. 有监督/无监督。
3. 线性模型/非线性模型。
4. 特征离散/特征连续。
5. 凸优化/非凸优化。

具体内容：

1. 二分类线性可分支持向量机
   1. 线性模型
      1. 定义：支持向量机（support vector machine, SVM）。
      2. 基本形式：有监督二分类线性分类模型。
      3. 扩展形式：
         1. 有监督二分类非线性分类模型。
         2. 有监督多分类（线性/非线性）分类模型。
         3. 有监督线性回归模型（support vector regression, SVR）
         4. 基于核函数的SVM/SVR。
   2. 最大间隔分类器。SVM和线性模型的区别。线性模型和SVM具体形式上都是通过一条线来对数据进行分类。那么形式上它们之间有什么区别呢？**其中核心思路就是：如何选择一条最合适的线来把这些数据分开？具体实现：最大间隔分类器，也就是说要求分类器距离数据最远。**。
      |![最大间隔分类器](../../pictures/SVMMaximumIntervalClassifier.jpg "最大间隔分类器")|
      |:--:|
      | *Figure 5.1* |
      1. 离数据最远的线性分类器最安全。
      2. 离数据最远的线性分类器最容易泛化。
      3. SVM模型式线性模型的一种。
      4. 支持向量的定义：**第一，分类器与最远的线性分类器平行；第二，该分类器穿过了具体最远线性分类器最近的数据。那么就称这些数据为支持向量**。
   3. SVM与逻辑回归的对比
      ||预测函数|损失函数|正则项|优化目标|
      |---|---|---|---|---|
      |逻辑回归|其预测函数之一是sigmod函数，实际输出的是一个概率。|其损失函数之一是最大似然函数/交叉熵损失函数。|正则项之一是$L_2$正则|损失函数和正则项的组合。凸优化，梯度下降。|
      |SVM|预测函数是$y=sgn(w^Tx)$。输出的不是概率，而是+1或者-1，代表分类。实际上就是一种特殊的非线性函数。预测数出和$w^tx$的绝对值大小无关，之和$w^tx$的符号有关。$w^tx$几何意义：正比于x到平面的有向距离。|优化函数：$l(x,y)=\frac{1}{n}\sum\limits_{i=1}^{n}max\{0,1-y_iw^Tx_i\}+\frac{\lambda}{2}\|w\|_2^2$。模型的要求：1. 模型形式：线性模型；2. 损失函数：Hinge损失函数； |正则项；$L_2$正则防止过拟合，最大化分类间隔。|凸优化，SMO算法。|

      ||逻辑回归|支持向量机|
      |---|---|---|
      |模型|二分类概率线性模型|二分类决策线性模型|
      |正则化|$L_2$正则化|$L_2$正则化|
      |损失函数|Logistic损失函数|Hinge损失函数|
      |原始优化问题|凸优化，梯度下降|凸优化，SMO算法|
2. 二分类线性不可分支持向量机
   1. 线性支持向量机的几何解释。
      1. 如何刻画/描述数据到分类器的间隔。$x_i$到直线$y=w^Tx$的距离是$\frac{|w^Tx|}{|w|_2}$。$y=w^Tx$在数据集上的间隔$min_i\frac{|w^Tx|}{|w|_2}$。注意到$\frac{|w^Tx|}{|w|_2}$关于w是齐次的，所以可以找到一个w，使得$min_i|w^Tx|=1$。
      2. 优化目标函数：如果所有点都被正确分类，那么：$y_iw^Tx_i\leqslant 1, \forall i \in [n]$。最大分类间隔$max\frac{1}{|w|_2} \Leftrightarrow min|w|_2^2$。非约束优化形式：$min\text{ }l(x,y)=\frac{1}{n}\sum\limits_{i=1}^{n}max\{0,1-y_iw^Tx_i\}+\frac{\lambda}{2}|w|_2^2$。两种形式等价。
   2. 松弛变量。
      1. **为了将这种方式引入到非线性的情况下（也就是说因为一条直线无法把两类数据分开，就需要一条曲线来将两个数据分开）**。这个时候就引入了松弛变量$\epsilon_i$。**$\epsilon_i$的作用就是允许数据在线性分类器$\epsilon_i$距离内不被正确的分类**。
      2. 几何意义：允许线性不可分的点到分类器的距离小于0。约束条件也就是$y_iw^Tx_i\leqslant 1-\epsilon_i, \epsilon_i \leqslant 0, \forall i \in [n]$。
   3. 线性不可分支持向量机。
      1. 线性可分与线性不可分是统一形式的。
      2. 在第一点的基础上，SVM是数据自适应的。
      3. 本质是一个凸优化问题，~~二次规划~~（二次规划还没有理解）。
      4. 可以转换为二次规划一般形式求解，也可以使用梯度下降法求解。
3. 多分类支持向量机。主要的思路是如何将一个多分类问题转化为二分类问题。
   1. one vs one方法。
      1. 思路：所有类别两两之间都建立分类器（是任意两个类别之间）。就是如果有n个类别，那么就需要建立$\frac{n(n-1)}{2}$个分类器。
      2. 优点：适用性广，LibSVM默认的实现方法。对于所有二类分类器都可以使用，概率/非概率分类器均可。
      3. 缺点：训练时计算复杂度为$O(k^2)$，测试时的计算复杂度为$O(k^2)$。也就是计算复杂度高。
   2. one vs all方法。
      1. 思路：任意一个类别相对其他所有其他类别之间建立分类器。如果有n个类别，那么就需要家里n个分类器。每个分类器专门来预测对应类别的分数。每次需要使用全部的分类器。
      2. 优点：计算复杂度低。训练时计算复杂度为$O(k)$，测试时的计算复杂度为$O(k)$。
      3. 缺点：适用性有限，多使用于概率分类器，例如逻辑回归需求中。
4. SVM工具包介绍。
   1. LibSVM，目前使用最广泛的。提供了命令行使用接口。对所有语言均有接口。是一个比较高效的实现。
   2. SVMLight，是C++实现的。基于SVM的结构预测和半监督的SVM，支持SVM排序的学习算法。
   3. Scikit-learn，python实现的、轻量级的、接口简单的库。支持线性、非线性、回归和分类数据需求。
5. SVM对偶形式
   1. SVM的对偶形式
      1. SVM约束优化问题。线性SVM的原问题（Primal Problem）是：目标是一个关于w的二次函数：$min\text{ }\frac{1}{2}|w|_2^2$，约束是关于w的线性函数：$s.t. \text{ }y_iw^Tx_i \leqslant 1, \forall i \in [n]$。核心是二次凸优化问题（Quadratic Programming）。光滑优化函数。局部最优值即全局最优值。
         1. 凸集合，用描述化的语言表述为：**对于集合中任意两点的连线依然在集合内部**。
            |![凸集合](../../pictures/SVMConvexOptimization.jpg "凸集合")|
            |:--:|
            | *Figure 5.2* |
         2. 凸函数，用描述化的语言表述为：**定义域中任意两点连线组成的线段都在这两点的函数曲线（面）上方**。
            |![凸函数](../../pictures/SVMConvexFunction.jpg "凸函数")|
            |:--:|
            | *Figure 5.3* |
      2. Lagrange乘子法。~~这个部分需要进一步的了解一下~~。
         |![Lagrange乘子法](../../pictures/LagrangeMultiplierMethod.jpg "Lagrange乘子法")|
         |:--:|
         | *Figure 5.4* |
      3. SVM的对偶形式
   2. 核函数以及核技巧
      1. 特征映射（这是非常重要的一个技巧）。**将输入数据从低维空间映射到高维空间的函数变换，使得变换后的数据更加容易（使用一个线性的关系）进行处理（分类/回归）**。
         |![特征由低维向高维映射](../../pictures/SVMFeatureMapping.jpg "特征由低维向高维映射")|
         |:--:|
         | *Figure 5.4* |
      2. 如何定义特征变换？**因为显式的定义特征变换显然会增加计算的复杂度**。比如原本是1000维的特征，通过一个二项式变换之后有了500,000个特征。为了解决这个问题引入了核函数，**其核心目的在于：我们不需要显式的计算特征映射，只关心的是变化后的特征的内积**。
         |![核函数](../../pictures/SVMCoreFunction.jpg "核函数")|
         |:--:|
         | *Figure 5.5* |
         ~~这个地方没有理解为什么只需要关心内积就可以了。~~**实际上核函数隐式的定义了特征映射的规则**。核函数的计算是在原空间，核函数的计算复杂度比较低。
      3. 几种不同的核函数，每一种核函数对应的是一种不同的映射模式。
         1. 线性核函数：$K(x, x')=x\cdot x'$。
         2. 拉普拉斯Laplacian核函数：$K(x, x')=exp(-\lambda|x-x'|)$。
         3. 高斯Gaussion核函数：$K(x, x')=exp(-\lambda|x-x'|^2)$。
         4. 多项式核函数：$K(x, x')(x\cdot x'+c)^k,k\in N)$。
         5. 条件密度核函数：$K(x, x')=E_c[p(x|c)\cdotp(x'|c)]$。
      4. 常用的核函数是高斯核函数和多项式核函数。
         1. **高斯核函数对应无穷维特征空间映射。在工程应用中，其中$\lambda$的选择至关重要**。
         2. **多项式核函数对应有限维特征空间映射。在工程应用中，其中指数k的选择至关重要**。
      5. 核技巧：将线性模型转换为非线性模型，将低维空间通过非线性映射到高维空间。这里需要看下面的部分做了解释：新模型在变化后的空间仍然是线性模型，新模型在原空间相对于x是非线性模型。
   3. 非线性支持向量机。如何将线性SVM扩展为非线性SVM？
      |![非线性映射的例子图示](../../pictures/SVMNolinearMapping.jpg "非线性映射的例子图示")| 
      |:--:|
      | *5.6* |
      1. 通过将内积替换成核函数即可。
      2. 新模型在变化后的空间仍然是线性模型。
      3. 新模型在原空间相对于x是非线性模型。
      4. 计算复杂度较小。只需要计算核矩阵$\boldsymbol{K}=\boldsymbol{K}_{ij}=\boldsymbol{K}(x_i,x_j)$。
         1. 非线性的优化问题和线性的优化问题几乎是一样的，不同的地方在于把核函数换掉了。
         2. 求凸二次规划问题。
         3. 可以求得全局最优解（~~这个地方有问题啊，如果是非凸函数如何求得全局最优解？~~）。
      5. 求解非线性支持向量机优化问题使用的算法是：SMO（Sequential minimal optimization, SMO）算法。SMO算法是Coordinate ascent算法的一个特例。另外一种算法是坐标上升法。
         1. 坐标上升算法
            1. 使用于光滑凸优化问题。
            2. 优化多个变量。
            3. 每次仅优化其中一个变量，固定其他所有变量不变，直至算法收敛。
            4. 目前SVM求解的最快算法，也是LibSVM的默认实现算法，通常远快于梯度下降算法。
         2. 坐标上升算法：目标$min \text{ }x_1^2+x_2^2$。步骤如下：
            1. 随机初始化算法$(x_1,x_2)=(-3, -4)$。
            2. 固定$x_2$，优化一维函数$x_1^2 + 16$
            3. 求得最优解为$x_1=0$
            4. 固定$x_1=0$，优化一维函数$0+x_2^2$
            5. 求得最优解$x_2=0$
            6. 再次固定$x_2=0$，优化一维函数$x_1^2+0$
            7. 求得最优解$x_1=0$，与上一轮迭代值相同
            8. 再次固定$x_1=0$，优化一维函数$0+x_2^2$
            9. 求得最优解$x_2=0$，与上一轮迭代值相同
            10. 算法收敛，停止，得到全局最优解$(x_1,x_2)=(0,0)$
   4. 支持向量回归（实例）

   ```python
   # ......
   from sklearn.svm import SVC
   from sklearn import datasets
   import matplotlib.pyplot as plt

   # 固定随机种子，保证结果复现。
   np.random.seed(42)

   # 绘图设置
   %matplotlib inline
   plt.rcParams['axes.labelsize'] = 14
   plt.rParams['xtick.labelsize'] = 12
   plt.rParams['ytick.labelsize'] = 12
   # ......

   iris = datasets.load_iris()
   X = iris["data"][:, (2, 3)] # petal length, petal width
   y = iris["target"]

   setosa_or_versicolor = (y==0) | (y==1)
   X = X[setosa_or_versicolor]
   y = y [setosa_or_versicolor]

   # SVM classifier model
   # 这里用的核函数是线性核函数，这里没有用高斯核和多项式核。
   svm_clf = SVC(kernel="linear", C=float("inf"))
   svm_clf.fit(X,y)

   # 注意C这个参数，这个是容错的范围。在实际工程环境中一定要有C这个参数。
   SVC(C=inf, kernel="linear")

   x0 = np.linspace(0, 5.5, 200)
   pred_1 = 5 * x0 - 20
   pred_2 = x0 - 1.8
   pred_3 = 0.1 * x0 + 0.5

   # 画出决策边界
   def plot_svc_decision_boundary(svm_clf, xmin, xmax):
      w = svm_clf.coef_[0]
      b = svm_clf.intercept_[0]

      # 决策边界 w0 * x0 + w1 * x1 + b = 0
      # => x1=-w0/w1 * x0 - b/w1
      x0 = np.linspace(xmin, xmax, 200)
      decision_boundary = -w[0]/w[1] * x0 - b/w[1]

      margin = 1/w[1]
      gutter_up = decision_boundary + margin
      gutter_down = decision_boundary - margin

      svs = svm_clf.support_vectors_
      plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')
      plt.plot(x0, decision_boundary, "k-", linewidth=2)
      plt.plot(x0, gutter_up, "k--", linewidth=2)
      plt.plot(x0, gutter_down, "k--", linewidth=2)
   
   plt.figure(figsize=(12, 2.7))

   plt.subplot(121)
   plt.plot(x0, pred_1, "g--", linewidth=2)
   plt.plot(x0, pred_2, "m-", linewidth=2)
   plt.plot(x0, pred_3, "r-", linewidth=2)
   plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris-Versicolor")
   plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris-Setosa")
   plt.xlabel("Petal length", fontsize=14)
   plt.ylabel("Petal width", fontsize=14)
   plt.legend(loc="upper left", fontsize=14)
   plt.axis([0, 5.5, 0, 2])

   plt.subplot(122)
   plot_svc_decision_boundary(svm_clf, 0, 5.5)
   plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs")
   plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo")
   plt.xlabel("Petal length", fontsize=14)
   plt.axis([0, 5.5, 0, 2])

   plt.savefig("large_margin_classification_plot")
   plt.show()
   ```

   1. 注意事项：
      1. SVM对特征幅度敏感，注意scaling（归一化）。这个的意思就是特征的多个维度的数据最好是在一个量级里面，这样做出的决策边界才比较合理。举例，比如$x_1$的范围是在$(100,1000)$这个范围，而$x_2$的范围是在$(0,10)$这个范围，那么在做决策的时候会更偏向于$x_1$做决策。这个时候需要将$x_1$做一个scaling缩小，或者将$x_2$放大。
      2. 对异常值非常敏感。SVM是期望将所有的值都涵盖在模型里面，也就是对所有的模型都能进行分类。由于在实际数据中难以避免存在异常值，所以采用了一种容错机制。看下面的第3点。
      3. 最大容错和间隔。可以理解为增加简单正则化的概念。

      ```python
      import numpy as np
      from sklearn import datasets
      from sklearn.pipeline import Pipeline
      from sklearn.preprocessing import StandardScaler
      from sklearn.svm import LinearSVC

      iris = datasets.load_iris()
      X = iris["data"][:, (2, 3)] # petal length, petal width
      y = (iris["target"] == 2).astype(np.float64) # Iris-Virginica

      # sklearn.pipeline的用法是将多个处理步骤组合成成一个流程，这样做了封装之后方便使用。[pipeline参考](https://juejin.cn/post/7029482491694022670)
      svm_clf = Pipeline([
                        ("scaler", StandardScaler()),
                        ("linear_svc", LinearSVC(C=1, loss="hinge", random_state=42))
                        ])
      svm_clf.fit(X, y)

      # 对数据进行预测。
      svm_clf.predict([[5.5, 1.7]])
      ```

      ```python
      # 不同的正则强度。
      scaler = StandardScaler()
      svm_clf1 = LinearSVC(C=1, loss="hinge", random_state=42)
      svm_clf2 = LinearSVC(C=100, loss="hinge", random_state=42)

      scaled_svm_clf1 = Pipeline([
                        ("scaler", scaler),
                        ("linear_svc", svm_clf1)
                        ])
      scaled_svm_clf2 = Pipeline([
                        ("scaler", scaler),
                        ("linear_svc", svm_clf2)
                        ])
      scaled_svm_clf1.fit(X, y)
      scaled_svm_clf2.fit(X, y)

      # Convert to unscaled parameters
      b1 = svm_clf1.decision_function([-scaler.mean_/scaler.scale_])
      b2 = svm_clf2.decision_function([-scaler.mean_/scaler.scale_])
      w1 = svm_clf1.coef_[0] / scaler.scale_
      w2 = svm_clf2.coef_[0] / scaler.scale_
      svm_clf1.intercept_ = np.array([b1])
      svm_clf2.intercept_ = np.array([b2])
      svm_clf1.coef_ = np.array([w1])
      svm_clf2.coef_ = np.array([w2])

      # Find support vectors (LinearSVC does not do this automatically)
      t = y * 2 - 1
      support_vectors_idx1 = (t * (x.dot(w1) + b1) < 1).ravel()
      support_vectors_idx2 = (t * (x.dot(w2) + b2) < 1).ravel()
      svm_clf1.support_vectors_ = X[support_vectors_idx1]
      svm_clf2.support_vectors_ = X[support_vectors_idx2]

      plt.figure(figsize=(12, 3.2))
      plt.subplot(121)
      plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^", label="Iris-Virginica")
      plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs", label="Iris-Versicolor")
      plot_svc_decision_boundary(svm_clf1, 4, 6)
      plt.xlabel("Petal length", fontsize=14)
      plt.ylabel("Petal width", fontsize=14)
      plt.legend(loc="upper left", fontsize=14)
      plt.title("$C={}$".format(svm_clf1.C), fontsize=16)
      plt.axis([4, 6, 0.8, 2.2])

      plt.subplot(121)
      plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^", label="Iris-Virginica")
      plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs", label="Iris-Versicolor")
      plot_svc_decision_boundary(svm_clf2, 4, 6)
      plt.xlabel("Petal length", fontsize=14)
      plt.legend(loc="upper left", fontsize=14)
      plt.title("$C={}$".format(svm_clf2.C), fontsize=16)
      plt.axis([4, 6, 0.8, 2.2])
      plt.savefig("regularization_plot")

      ```

   2. 非线性分类
      1. 非线性示例1

         ```python
         X1D = np.linspace(-4, 4, 9).reshape(-1, 1)
         X2D = np.c_[X1D, X1D**2]
         y= np.array([0, 0, 1, 1, 1, 1, 1, 0, 0])

         plt.figure(figsize=(11, 4))

         plt.subplot(121)
         plt.grid(True, which='both')
         plt.plot(X1D[:, 0][y==0], np.zeros(4), "bs")
         plt.plot(X1D[:, 0][y==1], np.zeros(5), "g^")
         plt.gca().get_yaxis().set_ticks([])
         plt.xlabel(r"$x_1$", fontsize=20)
         plt.axis([-4.5, 4.5, -0.2, 0.2])

         plt.subplot(122)
         plt.grid(True, which='both')
         plt.axhline(y=0, color='k')
         plt.axvline(x=0, color='k')
         plt.plot(X2D[:, 0][y==0], X2D[:, 1][y==0], "bs")
         plt.plot(X2D[:, 0][y==1], X2D[:, 1][y==1], "g^")
         plt.xlabel(r"$x_1$", fontsize=20)
         plt.ylabel(r"$x_2$", fontsize=20, rotation=0)
         plt.gca().get_yaxis().set_ticks([0, 4, 8, 12, 16])
         plt.plot([-4.5, 4.5], [6.5, 6.5], "r--", linewidth=3)
         plt.axis([-4.5, 4.5, -1, 17])

         plt.subplots_adjust(right=1)
         plt.savefig("higher_dimensions_plot", tight_layout=False)
         plt.show()
         ```

      2. 非线性示例2
         详见[非线性示例2](../../codes/4BasicKnowledgePoints/1SVMNonlinearClassification.ipynb)。

## 6. 贝叶斯算法

1. 朴素贝叶斯
   1. 条件概率公式。基本概念。
      1. 先验概率：根据以往经验和分析得到的概率。用P(Y)来代表在没有训练数据钱假设Y拥有的初始概率。
      2. 后验概率：根据已经发射的时间来分析得到的概率。以P(Y|X)代表假设X成立的情况下观察到Y数据的概率，因为它反映了在看到训练数据X后Y成立的置信度。
      3. 联合概率：联合概率是指在多元的概率分布中多个随机变量分别满足各自条件的概率。X与Y的联合概率表示为P(X, Y)、P(XY)或者$P(X \bigcap Y)$。
   2. 贝叶斯方程
      1. $P(Y|X)=\frac{P(X, Y)}{P(X)}=\frac{P(X|Y)P(Y)}{P(X)}$。其中$P(Y|X)$是后验概率。$P(X|Y)$是似然度。$P(Y)$是先验概率。$P(X)$是边际似然度。
      2. 朴素贝叶斯是典型的生成学习方法。生成方法有训练数据学习联合概率分布$P(X, Y)$，然后求得后验概率分布$P(Y|X)$。具体来说，**利用训练数据学习。$P(X|Y)$和$P(Y)$的估计，得到联合概率分布$P(X, Y)=P(X|Y)P(Y)$**。
      3. ~~如何理解呢？$P(Y|X)$其中Y是标签数据，X是训练数据，那么就是在观察到X之后Y的概率。P(Y)就是没有看到任何事件的情况下，Y的概率是多少~~。这个理解有点问题。
   3. 朴素贝叶斯的定义
      1. 设$x=\{a_1, a_2, \cdots , a_m\}$为一个待分类项，而每个$a_i$为x的一个特征属性。
      2. 有类别集合$C=\{y_1, y_2, \cdots , y_n\}$。
      3. 计算$P(y_1|x), P(y_2|x), \cdots, P(y_n|x)$。
      4. 如果$P(y_k|x)=max\{P(y_1|x), P(y_2|x), \cdots, P(y_n|x)\},x\in y_k$。~~这个地方有问题吧~~
   4. 朴素贝叶斯的例子。如图所示的例子。通过天气、湿度、风级来判断是否适合去打球。一共14条数据，其中5条是不适合打球，9条适合打球。在不知道天气、湿度、风级3个信息的情况下，能打球的的概率是$\frac{9}{14}$，这个概率就是先验概率。然后知道3个信息的情况下再计算是否能打球的概率，这就是后验概率。![朴素贝叶斯的例子](../../pictures/NaiveBayesExample.jpg "朴素贝叶斯的例子")
   5. 常见的应用场景：简单的文件的判断，垃圾邮件的判断等。
2. 贝叶斯网络和有向分离
   1. 贝叶斯网络(bayesian network)，又称为信念网络（Belief Network），或有向无环图模型（directed acyclic graphical model），是一种概率图模型。它是一种模拟人类推理过程中因果关系的不确定性处理模型，其网络拓扑结构是一个有向无环图（DAG）。贝叶斯网络的邮箱芜湖安图的节点表示随机变量${x_1, x_2, x_3, \cdots, x_n}$，它们可以是可观察到的变量，或隐变量、未知参数等。认为有因果关系（或非条件独立）的变量或明天则用箭头来连接。若两个节点间以一个单箭头连接在一起，表示其中一个节点是“因（parents）”，另一个是“果（children）”，两个节点就会产生一个条件概率值。总而言之，连接两个节点的箭头代表此两个随机变量是具有因果关系，或非条件独立。![贝叶斯网络](../../pictures/BayesianNetwork.jpg "贝叶斯网络")
   2. 有向分离（D-Separation）是一种用来判断变量是否条件独立的图形化方法。换而言之，对于一个DAG，有向分离方法可以快速的判断出两个节点之间是否是条件独立的。更具贝叶斯的3种形式来做讲解：
      1. head-to-head：在c未知的情况下，a、b被阻断（blocked），是独立的。称之为head-to-head条件独立。![head-to-head](../../pictures/BayesDSeparationHeadtoHead.jpg "head-to-head")
      2. tail-to-tail：需要考虑c已知和未知2种情况。c未知的时候，a,b不独立。c已知的时，a,b独立。所以在c给定的情况下，a,b被阻断，是独立的，称之为tail-to-tail条件独立。![tail-to-tail](../../pictures/BayesDSeparationTailtoTail.jpg "tail-to-tail")
      3. head-to-tail：要考虑c已知和未知2种情况。c未知的时候，a,b不独立。c已知的时，a,b独立。所以在c给定的情况下，a,b被阻断，是独立的，称之为head-to-tail条件独立。![head-to-tail](../../pictures/BayesDSeparationHeadtoTail.jpg "head-to-tail")
3. 实例。[参考代码](../../codes/4BasicKnowledgePoints/2NaiveBayesClassification.ipynb)。

## 7. 主题模型