# 第二次作业

## 1 验证逻辑回归损失函数的梯度的形式

证明：
$$
\begin{aligned}
& \nabla_\theta J(\theta)
 =-\nabla_\theta \sum \limits_{i=1}^{m} (y_i log(h_{\theta}(x_i))+(1-y_i)log(1-h_{\theta}(x_i))) \\
& \because \nabla \text{的对象是}\theta \\
& \therefore \\
& = -\sum \limits_{i=1}^{m} \nabla_\theta (y_i log(h_{\theta}(x_i))+(1-y_i)log(1-h_{\theta}(x_i))) \\
& = -\sum \limits_{i=1}^{m} (y_i \nabla_\theta log(h_{\theta}(x_i))+(1-y_i)\nabla_\theta log(1-h_{\theta}(x_i)))\\
& \text{其中：} \\
& \nabla_\theta log(h_{\theta}(x_i)) \\
& =\frac{1}{h_{\theta}(x_i)} \nabla_\theta (h_{\theta}(x_i)) \\
& =\frac{1}{h_{\theta}(x_i)} h_{\theta}(x_i)(1-h_{\theta}(x_i))x_i \\
& =\frac{h_{\theta}(x_i)(1-h_{\theta}(x_i))x_i}{h_{\theta}(x_i)} \\

& \nabla_\theta log(1-h_{\theta}(x_i)) \\
& =\frac{1}{1-h_{\theta}(x_i)} (-1) (h_{\theta}(x_i)(1-h_{\theta}(x_i)))x_i \\
& =-\frac{h_{\theta}(x_i)(1-h_{\theta}(x_i))x_i}{1-h_{\theta}(x_i)} \\
& \text{代回上式中：} \\
& = -\sum \limits_{i=1}^{m}(y_i \frac{h_{\theta}(x_i)(1-h_{\theta}(x_i))x_i}{h_{\theta}(x_i)} + (y_i -1)\frac{h_{\theta}(x_i)(1-h_{\theta}(x_i))x_i}{1-h_{\theta}(x_i)}) \\
& = -\sum \limits_{i=1}^{m}h_{\theta}(x_i)(1-h_{\theta}(x_i))x_i(\frac{y_i}{h_{\theta}(x_i)}+\frac{y_i -1}{1-h_{\theta}(x_i)}) \\
& = -\sum \limits_{i=1}^{m} h_{\theta}(x_i)(1-h_{\theta}(x_i))x_i\frac{y_i - y_ih_{\theta}(x_i) + y_i h_{\theta}(x_i) - h_{\theta}(x_i)}{h_{\theta}(x_i)(1-h_{\theta}(x_i))} \\
& = -\sum \limits_{i=1}^{m} x_i(y_i - h_{\theta}(x_i)) \\
& = \sum \limits_{i=1}^{m} x_i(h_{\theta}(x_i)) - y_i) \\
& \text{证毕}
\end{aligned}
$$  

## 2 证明矩阵是一个群

M是一个正交矩阵，而且ti是一个向量。需要正面M满足4个条件。
证明：
$$
\begin{aligned}
& \boldsymbol{M}_i = \begin{bmatrix}
&\boldsymbol{R}_i & \boldsymbol{t}_i \\
&\boldsymbol{0} &\boldsymbol{1}
\end{bmatrix}\in \mathbb{R}, \boldsymbol{R}_i \in \mathbb{R^{3\times 3}}(det(\boldsymbol{R}_i)=1), \text{and} \quad \boldsymbol{t}_i \in \mathbb{R^{3\times 1}}\\

& \text{1. proof closure:} \\
& \text{set} \boldsymbol{M}_j \in \boldsymbol{M}_i, \boldsymbol{M}_k \in \boldsymbol{M}_i, \text{and} \quad \boldsymbol{R}_j \in \boldsymbol{R}_i, \boldsymbol{R}_k \in \boldsymbol{R}_i, \boldsymbol{t}_j \in \boldsymbol{t}_i, \boldsymbol{t}_k \in \boldsymbol{t}_i. \\
& M_j \cdot M_k = \begin{bmatrix}
&\boldsymbol{R}_j & \boldsymbol{t}_j \\
&\boldsymbol{0} &\boldsymbol{1}
\end{bmatrix} \cdot \begin{bmatrix}
&\boldsymbol{R}_k & \boldsymbol{t}_k \\
&\boldsymbol{0} &\boldsymbol{1}
\end{bmatrix} \\
& =\begin{bmatrix}
&\boldsymbol{R}_j \boldsymbol{R}_k & \boldsymbol{R}_j\boldsymbol{t}_k + \boldsymbol{t}_j\\
&\boldsymbol{0} &\boldsymbol{1}
\end{bmatrix} \\
& \because \boldsymbol{R}_i \in \mathbb{R^{3\times 3}}, \quad \text{实数之间四则运算的结果肯定还是属于实数的.}\\
& \therefore \boldsymbol{R}_j \boldsymbol{R}_k \in \mathbb{R^{3\times 3}}.\\
& \text{同理可得：}\\
& \because \boldsymbol{t}_i \in \mathbb{R^{3\times 1}}，\\
& \therefore \boldsymbol{R}_j\boldsymbol{t}_k + \boldsymbol{t}_j \in \mathbb{R^{3\times 1}}.\\
& \therefore \boldsymbol{M}_j \cdot \boldsymbol{M}_k \in \mathbb{R^{3\times 3}}.\\

& \text{2. proof associativity:} \\
& \text{set} \boldsymbol{M}_l \in \boldsymbol{M}_i,\text{and} \quad \boldsymbol{R}_l \in \boldsymbol{R}_i, \boldsymbol{t}_l \in \boldsymbol{t}_i. \\
& (\boldsymbol{M}_j \cdot \boldsymbol{M}_k )\cdot \boldsymbol{M}_l= \begin{bmatrix}
&\boldsymbol{R}_j \boldsymbol{R}_k & \boldsymbol{R}_j\boldsymbol{t}_k + \boldsymbol{t}_j\\
&\boldsymbol{0} &\boldsymbol{1}
\end{bmatrix} \cdot \begin{bmatrix}
&\boldsymbol{R}_l & \boldsymbol{t}_l \\
&\boldsymbol{0} &\boldsymbol{1}
\end{bmatrix}\\
& =\begin{bmatrix}
&\boldsymbol{R}_j \boldsymbol{R}_k \boldsymbol{R}_l & \boldsymbol{R}_j \boldsymbol{R}_k\boldsymbol{t}_l + \boldsymbol{R}_j\boldsymbol{t}_k + \boldsymbol{t}_j\\
&\boldsymbol{0} &\boldsymbol{1}
\end{bmatrix} \\
& \boldsymbol{M}_j \cdot (\boldsymbol{M}_k \cdot \boldsymbol{M}_l)= \begin{bmatrix}
&\boldsymbol{R}_j & \boldsymbol{t}_j \\
&\boldsymbol{0} &\boldsymbol{1}
\end{bmatrix}  \cdot \begin{bmatrix}
&\boldsymbol{R}_k \boldsymbol{R}_l & \boldsymbol{R}_k\boldsymbol{t}_l + \boldsymbol{t}_k\\
&\boldsymbol{0} &\boldsymbol{1}
\end{bmatrix} \\
& = \begin{bmatrix}
&\boldsymbol{R}_j \boldsymbol{R}_k \boldsymbol{R}_l & \boldsymbol{R}_j \boldsymbol{R}_k\boldsymbol{t}_l + \boldsymbol{R}_j\boldsymbol{t}_k + \boldsymbol{t}_j\\
&\boldsymbol{0} &\boldsymbol{1}
\end{bmatrix} \\
& \therefore (\boldsymbol{M}_j \cdot \boldsymbol{M}_k )\cdot \boldsymbol{M}_l=\boldsymbol{M}_j \cdot (\boldsymbol{M}_k \cdot \boldsymbol{M}_l) \\

& \text{3. proof identity:}\\
& \text{显然} \boldsymbol{E} \in \mathbb{R^{4\times 4}} \\
& \text{使得} \boldsymbol{E} \boldsymbol{M}_i = \boldsymbol{M}_i \boldsymbol{E} \\
& \therefore \exists \boldsymbol{I}. \\

& \text{4. proof inverse:} \\
& \because det(\boldsymbol{R}_i)=1 \\
& \therefore \boldsymbol{R}_i \text{可逆}. \\
& det(\boldsymbol{M}_i)=\begin{vmatrix}
&\boldsymbol{R}_i & \boldsymbol{t}_i \\
&\boldsymbol{0} &\boldsymbol{1}
\end{vmatrix}\\
& = det(\boldsymbol{R}_i) det(\boldsymbol{1})\\
& \because \boldsymbol{1} \in \mathbb{R^{1\times 1}} \\
& \therefore det(\boldsymbol{M}_i)= 1 \not= 0 \\
& \therefore \boldsymbol{M}_i \text{可逆}. \\
& \text{由矩阵的性质可得：} \boldsymbol{M}_i^{-1}\cdot \boldsymbol{M}_i=\boldsymbol{M}_i \cdot \boldsymbol{M}_i^{-1} = \boldsymbol{E}\\

& \text{综上所述：} \boldsymbol{M}_i \text{is a group.}
\end{aligned}
$$
