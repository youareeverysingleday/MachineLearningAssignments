# translate翻译

1. 题目：Learning Graph-based POI Embedding for Location-based Recommendation
将基于学习图POE嵌入应用于基于位置的推荐系统
2. 选择这两篇是因为在Survey on user location prediction based on geo-social networking data中表明这一篇是对于小时级的位置预测的进行了阐述。

|number|title of paper|internet source|local source|correlative field|illustration|
|---|---|---|---|---|---|
|1|Learning Graph-based POI Embedding for Location-based Recommendation|<https://Sci-hub.ee>|/|location prediction|English translate into chinese|

## 需要做的事情

## 重点语句

## 摘要

|编号|英语|中文|理解|
|---|---|---|---|
|1|extreme||/|
|2|sparsity||/|
|3|severe||/|
|4|degrade||/|
|5|spatiotemporal||/|
|6|context awareness|上下文感知，环境感知|/|
|7|address|有设法解决的意思|/|
|8|recent advances|研究进展|/|
|9|jointly|副词，统一地|/|
|10|sequential effect|序列作用|/|
|11|influence|名词，影响|/|
|12|temporal cyclic effect|时间周期性||
|13|semantic effect|语义作用||
|14|novel|形容词，新颖的。做名词是小说的意思||
||experimental|||
||superiority|名词，优势||
|||||
|||||

伴随着移动设备的迅猛增长和基于位置的社交网络的快速发展，基于位置的推荐变得非常重要了，它可以帮助人们发现有吸引力和兴趣的POI。尽管，极度稀疏的user-POI矩阵和冷启动问题导致了非常严峻的挑战。导致基于CF（这应该是协同过滤）的方法的推荐性能被显著的降低了。此外，基于位置的推荐需要在一个实时环境中时空上下文感知和用户最新偏好动态跟踪。

为了解决这些挑战，我们依赖在嵌入学习技术的最新进展，并且提出了一个通用的基于图的嵌入模型，在本文中称其为GE。GE通过将四个对应的关系图（POI-POI、POI-地域、POI-时间和POI-语义）嵌入到一个共同的低维空间中，然后以统一的方式来捕获顺序作用、地理影响、时间周期性作用和语义作用。然后，去支持实时推荐，我们开发了一个新颖的时间衰减方法，根据用户在隐性空间中学习的check-in POI嵌入来动态的计算用户的最新偏好。我们在两个真实的大规模数据集上进行了大量的实验来评估我们的模型性能，实验救过表明它由于其他竞争对手，尤其是在POI推荐冷启动方面。除此之外，我们研究了每个特征对于提高基于位置的推荐系统的贡献，**并且发现顺序作用和时间周期性作用比地理影响和语义作用更为重要**。

## 1. 简介

|编号|英语|中文|理解|
|---|---|---|---|
|1|acquisition|名词，采集|/|
|2|promptly|副词，迅速地|/|
|3|utilize|动词，利用，使用|/|
|4|facilitate|动词，促进，方便|/|
|5|concern|名词，担忧，考虑|/|
|6|plague|动词，困扰，折磨|/|
|7|||/|
|8|||/|
|9|||/|
|10|||/|
|||||

随着web 2.0、位置采集和无线通信技术的迅速发展，大量基于位置的社交网络在最近几年涌现了出来，比如Foursquare、Facebook Places、Gowalla和Loopt，在它们上面用户可以在POIs上check in，比如：商店、餐馆、景观。并且可以通过移动设备快速的在分享他们的在物理世界中的生活经历。它是最重要的能够利用用户check-in数据来产生个性化推荐的实时方式，它帮助用户去发现新的POI并且发现新的地域（比如：城市），从而方便广告商向目标用户发布移动广告。

不同于传统的桌面推荐系统推送“数字化的”信息，比如：电源推荐、音乐推荐等，基于位置的推荐系统典型的涉及移动用户和“物理”实体（比如景点），同样的也要经受更多挑战（就是风险与机遇并存）。

1. 数据稀疏。为了了解并且评估一个POI，用户必须通过物理上的方式访问POI，因此，相比于在线电源他们需要花费更多的代价来对评价一部电源。即使如果一个用户尝试访问了一个POI，他也可能基于隐私
和安全的考虑而不去check-in。因此，在LBSN中产生的用户check-in数据相比于对电源和音乐的评价数据而言是非常稀疏的。这个问题困扰着现在绝大多数的协同过滤推荐系统。

2. 上下文感知







|编号|英语|中文|理解|
|---|---|---|---|
|1|||/|
|2|||/|
|3|||/|
|4|||/|
|5|||/|
|6|||/|
|7|||/|
|8|||/|
|9|||/|
|10|||/|
|||||