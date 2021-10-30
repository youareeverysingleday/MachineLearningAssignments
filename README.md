# Machine Learning Assignments

study Machine Learning Assignments

## 1. first Machine Learning Assignment 

### 1.1. implement AdaBoost

1. completed is 20211022.
2. the choose error rate function is very important, because it direct decides how choose weak classifier. pay a attention, this code doesn't implement adaboost according to original design. but modify loop stop condition that error rate >=0.5. change the stop condition to global minimum error rate. maybe is wrong.

### 1.2. mathematics

1. problem is :

(Mathematics) There are n-dimensional data points and we can stack them into a data matrix: $X={x_i}_{i=1}^n, x_i\in R^{p\times 1}, X\in R^{p\times n}$.

The convariance matrix of $X$ is $C=\frac{1}{n-1}\sum\limits_{i=1}^{n}(x_i-\mu)(x_i-\mu)^T$, where$\mu=\frac{1}{n}\sum\limits_{i=1}^{n}x_i$(actually, it is the mean of the data points).

Based on discussions in our lecture, we know that if $\alpha_1$ is the eigen-vector associated with the largest eigen-value of $C$, the data projections along $\alpha_1$ will have the largest variance.

Now let's consider such an orientation $\alpha_2$. it is orthogonal to $\alpha_1$, the variance of data projections to $\alpha_2$ is the largest one.

Please prove that: $\alpha_2$ actually is the eigen-vector of $C$ associated to $C$'s second largest eigen-value. (We can assume that $\alpha_2$ is a unit vector.)

## 2. translation

add the part of translation.
|number|title of paper|internet source|local source|correlative field|illustration|
|---|---|---|---|---|---|
|1|neural collaborative filtering|http://staff.ustc.edu.cn/~hexn/papers/www17-ncf.pdf|./references/1NeuralCollaborativeFiltering.pdf|recommonder system|English translate into chinese|
|2||||||
