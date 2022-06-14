
[TOC]

# 常用Latex符号或举例

## 1. 参考

|编号|参考|
|---|---|
|1|手册，重要参考<https://www.zybuluo.com/codeep/note/163962>|
|2|一些重要字符速查<https://blog.csdn.net/katherine_hsr/article/details/79179622>|
|3|一些复杂矩阵的书写<https://zhuanlan.zhihu.com/p/266267223?utm_source=wechat_session>|

## 2. 常用符号速查表

### 2.1 希腊字母和希伯来文字母 Greek and Hebrew letters

|编号|符号|Latex代码|描述|
|---|---|---|---|
||$\alpha$|\alpha|各种希腊字母|
||$\beta$|\beta||
||$\gamma$|\gamma||
||$\Gamma$|\Gamma||
||$\delta$|\delta||
||$\Delta$|\Delta||
||$\epsilon$|\epsilon||
||$\varepsilon$|\varepsilon||
||$\zeta$|\zeta||
||$\eta$|\eta||
||$\theta$|\theta||
||$\Theta$|\Theta||
||$\vartheta$|\vartheta||
||$\iota$|\iota||
||$\pi$|\pi||
||$\phi$|\phi||
||$\Phi$|\Phi||
||$\psi$|\psi||
||$\Psi$|\Psi||
||$\omega$|\omega||
||$\Omega$|\Omega||
||$\chi$|\chi||
||$\rho$|\rho||
||$\omicron$|\omicron||
||$\sigma$|\sigma||
||$\Sigma$|\Sigma||
||$\nu$|\nu||
||$\xi$|\xi||
||$\Xi$|\Xi||
||$\tau$|\tau||
||$\lambda$|\lambda||
||$\Lambda$|\Lambda||
||$\mu$|\mu||
||$\Upsilon$|\Upsilon||
||$\partial$|\partial||
||$\aleph$|\aleph||
||$\beth$|\beth||
||$\daleth$|daleth||
||$\gimel$|\gimel||
||$$|||

|编号|符号|Latex代码|描述|
|---|---|---|---|
|1|$\sum\limits_{i=0}^n$|\sum \limits_{i=0}^n|求和|
|2|$\sum_{i=0}^n$|\sum_{i=0}^n||
||$\frac{\partial x}{\partial y}$|\frac{\partial x}{\partial y}|偏导数|
||$180^\circ$|180^\circ|表示摄氏度|
||$\bigodot$|\bigodot||
||$\pm$|\pm|正负号|
||$\mp$|\mp|正负号|
||$\lfloor$|\lfloor|下界|
||$\rfloor$|\rfloor|下界|
||$\lceil$|\lceil|上界|
||$\rceil$|\rceil|上界|
||$\prod$|\prod|连乘|
||$\prod \limits_{i=1}^n$|\prod \limits_{i=1}^n|连乘示例|
||$\coprod$|\coprod|N元余积|
||$\cdots$|\cdots|3个点，表示省略|
||$\dot{a}$|\dot{a}|一阶导数，变量上加一个点|
||$\ddot{a}$|\ddot{a}|二阶导数，变量上加两个点|
||$\int$|\int|积分|
||$\int_0^1 {x^2} \,{\rm d}x$|\int_0^1 {x^2} \,{\rm d}x|积分|
||$\iint$|\iint|双重积分|
||$\oint$|\oint|曲线积分|
||$\because$|\because|因为|
||$\therefore$|\therefore|所以|
||$\lbrace \rbrace$|\lbrace \rbrace||
||$\hat{y}$|\hat{y}|加帽，一般表示期望|
||$\check{y}$|\check{y}||
||$\breve{y}$|\breve{y}||
||$\overline{a+b+c+d}$|\overline{a+b+c+d}|上面加线，表示补集或者表示平均值|
||$\overline{A}$|\overline{A}|表示补集或均值|
||$\underline{a+b+c+d}$|\underline{a+b+c+d}|下面加线|
||$\overbrace{a+\underbrace{b+c}_{1.0}+d}^{2.0}$|\overbrace{a+\underbrace{b+c}_{1.0}+d}^{2.0}|上下加括号，并填写对应的值|
||$\vec{a}$|\vec{a}|向量，变量上加箭头|
||$\widehat{a}$|\widehat{a}|(线性回归，直线方程) a尖，变量上加上三角|
||$\widetilde{a}$|\widetilde{a}|颚化符号  等价无穷小，变量上加波浪线|
||$\boldsymbol{X}$|\boldsymbol{X}|加粗，表示向量或者矩阵|
||$\mathbf{X}$|\mathbf{X}|加粗，表示向量或者矩阵|
||$\pmb{C}$|\pmb{C}|加粗|
||$\mathbb{N}$|\mathbb{N}|自然数集|
||$\mathbb{Z}$|\mathbb{Z}|整数集|
||$\mathbb{Q}$|\mathbb{Q}|有理数|
||$\mathbb{R}^{m\times n}$|\mathbb{R}^{m\times n}|实数集|
||$\mathbb{A}$|\mathbb{A}|无理集|
||$\mathbb{C}$|\mathbb{C}|复数集|
||$\sqrt{2}$|\sqrt{2}|开方|
||$1\quad 2$|\quad|空格|
||$1\qquad 2$|\qquad|空2个格|
||$1\text{   } 2$|\text{   }|空3个格，**这种无法实现**|
||$1\; 2$|\;|空半格|
||$1\, 2$|\,|空半格|
||$\overleftarrow{xy}$|\overleftarrow{xy}|上方添加左箭头|
||$\overleftrightarrow{xy}$|\overleftrightarrow{xy}|上方添加双向箭头|
||$\rm{Word Style}$|\rm{Word Style}|罗马体|
||$\it{Word Style}$|\it{Word Style}|斜体|
||$\bf{Word Style}$|\bf{Word Style}|粗体|
||$\sf{Word Style}$|\sf{Word Style}|等线体|
||$\tt{Word Style}$|\tt{Word Style}|打字机体|
||$\frak{Word Style}$|\frak{Word Style}|旧德式字体|
||$\mathcal{WORD STYLE}$|\mathcal{Word Style}|花体（数学符号等），**仅大写可用**|
||$\mathbb{WORD STYLE}$|\mathbb{Word Style}|黑板粗体（定义域等），**仅大写可用**|
||$\mit{WORD STYLE}$|\mit{Word Style}|数学斜体，不支持，**仅大写可用**|
||$\scr{WORD STYLE}$|\scr{Word Style}|手写体，不支持，**仅大写可用**|
||$\boldsymbol{D}$|\boldsymbol{D}|**表示向量或者矩阵的加粗斜体**|
||$\color{red}{Latex}$|\color{red}{Latex}|给文字添加颜色，相关颜色详见下面的例子|
||$A \overset{some sentence}{=} B$|\overset{some sentence}{=}|文字在等号上面|
||$A \underset{some sentence}{=} B$|\underset{some sentence}{=}|文字在等号下面|
||$$|||

### 2.5 标准函数名 standard function names

|编号|符号|Latex代码|描述|
|---|---|---|---|
||$\arccos$|\arccos|反余弦|
||$\arcctg$|\arcctg|反余切|
||$\arcsin$|\arcsin|反正弦|
||$\arctan$|\arctan|反正切|
||$\cos$|\cos|余弦|
||$\csc$|\csc|余割|
||$\ctg$|\ctg|余切|
||$\exp$|\exp|以e为底的指数|
||$\lg$|\lg|以10为底的对数|
||$\lim$|\lim|求极限|
||$\ln$|\ln|以e为底的对数|
||$\log$|\log|对数|
||$\min$|\min|求最小值|
||$\max$|\max|求最大值|
||$\Pr$|\Pr|求概率|
||$\sin$|\sin|正弦|
||$\tan$|\tan|正切|
||$\log_{a}{x}$|\log{x}|以a为底的对数|
||$\ln{x}$|\ln{x}|以e为底的对数|
||$\lg{x}$|\lg{x}|以10为底的对数|
||$\sin\theta$|\sin\theta|正弦|
||$\cos\theta$|\cos\theta|余弦|
||$\tan\theta$|\tan\theta|正切|
||$\cot\theta$|\cot\theta|余切|
||$\arctan\theta$|\tan\theta|反正切|
||$\arcsin\frac{L}{r}$|\arcsin\frac{L}{r}|反正弦|
||$\max H$|\max H|最大值|
||$\log_\alpha x$|\log_\alpha x|以$\alpha$为底的对数|
||$\gcd(T,U,V,W,X)$|\gcd(T,U,V,W,X)||
||$\arg x$|\arg x||
||$a \bmod b$|a \bmod b|求余|
||$\lim \limits_{\rho \rightarrow 0} \frac{\Delta x}{\Delta y}$|\lim \limits_{\rho \rightarrow 0} \frac{\Delta x}{\Delta y}|极限|
||$\lim_{t\to n}T$|\lim_{t\to n}T|极限|
||$$|||

### 2.6 逻辑运算符或者关系运算法 binary operation/ relation symbols

|编号|符号|Latex代码|描述|
|---|---|---|---|
||$\wedge$|\wedge|逻辑且|
||$\vee$|\vee|逻辑或|
||$\mid$|\mid|竖线|
||$\bigoplus$|\bigoplus|异或|
||$\geqslant$|\geqslant|大于等于|
||$\leqslant$|\leqslant|小于等于|
||$\leq$|\leq|大于等于|
||$\geq$|\geq|小于等于|
||$\neq$|\neq|不等于|
||$\equiv$|\equiv|恒等|
||$\coloneqq$|\coloneqq|定义|
||$\not ={}$|\not ={}|不等于|
||$\approx$|\approx|约等于|
||$\sim$|\sim|约等于|
||$\propto$|\propto|正比于|
||$\widehat{=}$|\widehat{=}|相关于|
||$\cup$|\cup|并集|
||$\cap$|\cap|交集|
||$\bigcap$|\bigcap|交集|
||$\bigvee$|\bigvee|逻辑或|
||$\bigwedge$|\bigwedge|逻辑与|
||$\biguplus$|\biguplus|多重集|
||$\bigsqcup$|\bigsqcup||
||$\sim$|\sim||
||$\backsim$|\backsim||
||$\cong$|\cong||
||$\not>$|\not>|不大于|
||$\not<$|\not<|不小于|
||$\in$|\in|属于|
||$\ni$|\ni|属于|
||$\notin$|\notin|不属于|
||$\not\ni$|\not\ni|不属于|
||$\emptyset$|\emptyset|空集|
||$\subset$|\subset|子集|
||$\not\subset$|\not\subset|非子集|
||$\subseteq$|\subseteq|真子集|
||$\supseteq$|\supseteq||
||$\top$|\top||
||$\bot$|\bot|两个几何概念互相垂直；或表示两个事件互相独立|
||$\times$|\times|一般乘号、乘法、叉|
||$\cdot$|\cdot|点乘|
||$\ast$|\ast|乘号|
||$\circ$|\circ||
||$\bigotimes$|\bigotimes|克罗内克积|
||$\div$|\div|一般除号|
||$\propto$|\propto|正比于|
||$\triangleq$|\triangleq||
||$\bigcirc$|\bigcirc|圆圈|
||$$|||

### 2.7 箭头符号 arrorw symbols

|编号|符号|Latex代码|描述|
|---|---|---|---|
||$\Uparrow$|\uparrow|各种箭头|
||$\Uparrow$|\Uparrow||
||$\downarrow$|\downarrow||
||$\Downarrow$|\Downarrow||
||$\leftarrow$|\leftarrow||
||$\Leftarrow$|\Leftarrow||
||$\rightarrow$|\rightarrow||
||$\Rightarrow$|\Rightarrow|推出|
||$\updownarrow$|\updownarrow||
||$\Updownarrow$|\Updownarrow||
||$\leftrightarrow$|\leftrightarrow||
||$\Leftrightarrow$|\Leftrightarrow||
||$\Longleftrightarrow$|\Longleftrightarrow||
||$\Longleftarrow$|\Longleftarrow||
||$\longleftarrow$|\longleftarrow||
||$\longrightarrow$|\longrightarrow||
||$\Longrightarrow$|\Longrightarrow||
||$\mapsto$|\mapsto||
||$\longmapsto$|\longmapsto||
||$\nRightarrow$|\nRightarrow|不能推出|
||$\nLeftarrow$|\nLeftarrow||
||$\nLeftrightarrow$|\nLeftrightarrow||
||$$|||

### 2.8 杂项符号 miscellaneous symbols

|编号|符号|Latex代码|描述|
|---|---|---|---|
||$\forall$|\forall|任意|
||$\exists$|\exists|存在|
||$\nexists$|\nexists|不存在|
||$\complement$|\complement||
||$\angle A$|\angle A|角A|
||$\measuredangle A$|\measuredangle A|测量角|
||$\surd$|\surd|勾、正确的|
||$\square$|\square|方块|
||$\varnothing$|\varnothing|空变量|
||$\emptyset$|\emptyset|空集|
||$\ddots$|\ddots|斜着三点，注意没有反斜线的三点|
||$\cdots$|\cdots|横着三点|
||$\vdots$|\vdots|竖着三点|
||$\ldots$|\ldots|横着三点|
||$\infty$|\infty|无穷大|
||$\nabla$|\nabla|梯度|
||$$|||

## 3. 重要例子和不方便在表格中显示的公式

1. 更改文字颜色，将\\color{}大括号中的颜色替换即可。：
    代码：\color{red}{Latex \quad color \quad of \quad word}

    |编号|符号|Latex代码|说明|
    |---|---|---|---|
    |1|$\color{black}{Latex \quad color \quad of \quad word}$|black||
    |2|$\color{silver}{Latex \quad color \quad of \quad word}$|silver||
    |3|$\color{maroon}{Latex \quad color \quad of \quad word}$|maroon||
    |4|$\color{yellow}{Latex \quad color \quad of \quad word}$|yellow||
    |5|$\color{olive}{Latex \quad color \quad of \quad word}$|olive||
    |6|$\color{teal}{Latex \quad color \quad of \quad word}$|teal||
    |7|$\color{blue}{Latex \quad color \quad of \quad word}$|blue||
    |8|$\color{purple}{Latex \quad color \quad of \quad word}$|purple||
    |9|$\color{grey}{Latex \quad color \quad of \quad word}$|grey||
    |10|$\color{white}{Latex \quad color \quad of \quad word}$|white||
    |11|$\color{red}{Latex \quad color \quad of \quad word}$|red||
    |12|$\color{lime}{Latex \quad color \quad of \quad word}$|lime||
    |13|$\color{green}{Latex \quad color \quad of \quad word}$|green||
    |14|$\color{auqa}{Latex \quad color \quad of \quad word}$|auqa||
    |15|$\color{navy}{Latex \quad color \quad of \quad word}$|navy||
    |16|$\color{fuchsia}{Latex \quad color \quad of \quad word}$|fuchsia||

2. 在\text{}中可以再插入公式。**这种写法可以在行间公式中使用，在行内公式中无法正确显示**。
$$f(n)= \begin{cases} n/2, & \text {if $n$ is even} \\ 3n+1, & \text{if $n$ is odd} \end{cases}
$$

    代码：f(n)= \begin{cases} n/2, & \text {if $n$ is even} \\ 3n+1, & \text{if $n$ is odd} \end{cases}

3. 公式编号，**注意只能在行间公式中使用，不能在行内公式中使用**。
$$x+y = z \tag{1.1}$$
    代码：x+y = z \tag{1.1}

4. 最常用的多行公式或者证明过程可以使用：
$$
    \begin{aligned}
    & \text{不失一般性的设某个子群的2个陪集分别是：}g_1H, g_2H \\
    & \because g_1H, g_2H\text{之间存在交集，不失一般性的设两个陪集之间存在一个公共元素为}g_1h_1 = g_2h_2 \\
    & \Rightarrow g_1 = g_2h_2h_1^{-1} \\
    & \text{令：}h_3 = h_2h_1^{-1} \\
    & g_1H = g_2h_3H \\
    & \because \text{群的封闭性} \\
    & \therefore h_3H \in H \\
    & g_1H = g_2h_3H = g_2H \\
    & \text{两个陪集完全相等。}
    \end{aligned}
$$
    代码：
    \begin{aligned}
    & \text{不失一般性的设某个子群的2个陪集分别是：}g_1H, g_2H \\
    & \because g_1H, g_2H\text{之间存在交集，不失一般性的设两个陪集之间存在一个公共元素为}g_1h_1 = g_2h_2 \\
    & \Rightarrow g_1 = g_2h_2h_1^{-1} \\
    & \text{令：}h_3 = h_2h_1^{-1} \\
    & g_1H = g_2h_3H \\
    & \because \text{群的封闭性} \\
    & \therefore h_3H \in H \\
    & g_1H = g_2h_3H = g_2H \\
    & \text{两个陪集完全相等。}
    \end{aligned}

5. 数据表，没有具体含义：
$$
\begin{matrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{n1} & a_{n2} & \cdots & a_{nn}
\end{matrix}
$$

    代码：
    \begin{matrix}
    a_{11} & a_{12} & \cdots & a_{1n} \\
    a_{21} & a_{22} & \cdots & a_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{n1} & a_{n2} & \cdots & a_{nn}
    \end{matrix}
6. 注意在{array}后面有个{l}是必须的。
$$
\begin{array}{l}
a & b \\
c & d
\end{array}$$
    代码：
        \begin{array}{l}
        a & b \\
        c & d
        \end{array}
7. 行列式：
    $$
    \boldsymbol{D}=\begin{vmatrix}
        a_{11} & \cdots & a_{1k} &   &   &   \\
        \vdots &   & \vdots &   & 0 &   \\
        a_{k1} & \cdots & a_{kk} &   &   &   \\
        c_{11} & \cdots & c_{1k} & b_{11} & \cdots & b_{1n}\\
        \vdots &   & \vdots & \vdots &  & \vdots\\
        c_{n1} & \cdots & c_{nk} & b_{n1} & \cdots & b_{nn}
        \end{vmatrix}
    $$
        代码：
        \boldsymbol{D}=\begin{vmatrix}
        a_{11} & \cdots & a_{1k} &   &   &   \\
        \vdots &   & \vdots &   & 0 &   \\
        a_{k1} & \cdots & a_{kk} &   &   &   \\
        c_{11} & \cdots & c_{1k} & b_{11} & \cdots & b_{1n}\\
        \vdots &   & \vdots & \vdots &  & \vdots\\
        c_{n1} & \cdots & c_{nk} & b_{n1} & \cdots & b_{nn}
        \end{vmatrix}

    $$\boldsymbol{D}=\begin{vmatrix}
        a_{11} & a_{12} & \cdots & a_{1n} \\
        \vdots & \vdots & \ddots & \vdots \\
        a^{'}_{i1} & a^{'}_{i2} & \cdots & a^{'}_{in} \\
        \vdots & \vdots & \ddots & \vdots \\
        a_{n1} & a_{n2} & \cdots & a_{nn}
        \end{vmatrix}
    $$
        代码：
        \boldsymbol{D}=\begin{vmatrix}
        a_{11} & a_{12} & \cdots & a_{1n} \\
        \vdots & \vdots & \ddots & \vdots \\
        a^{'}_{i1} & a^{'}_{i2} & \cdots & a^{'}_{in} \\
        \vdots & \vdots & \ddots & \vdots \\
        a_{n1} & a_{n2} & \cdots & a_{nn}
        \end{vmatrix}

8. 矩阵：
$$\boldsymbol{D}=
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix}$$
    代码：
        \boldsymbol{D}=
        \begin{bmatrix}
        a & b \\
        c & d
        \end{bmatrix}

9. 分段公式：
$$L(Y,f(x))=
\begin{cases}
1, Y!=f(x) \\
0, Y = f(x)
\end{cases}$$
    代码：
        L(Y,f(x))=
        \begin{cases}
        1, Y!=f(x) \\
        0, Y = f(x)
        \end{cases}

10. 多行公式，[参考](https://blog.csdn.net/hu3350261/article/details/104902011)。**注意这个equation是带编号的，如果不需要带编号的直接使用aligned就可以了**。：

$$
\begin{equation}
\begin{aligned}
    &\dot{\boldsymbol{x}}=A \boldsymbol{x}+B \boldsymbol{u}
    , \quad 
    \boldsymbol{x}(0)=\boldsymbol{x}_{0}\\
    &y=C x+D u
\end{aligned}
\end{equation}
$$
    代码:
    \begin{equation}
    \begin{aligned}
        &\dot{\boldsymbol{x}}=A \boldsymbol{x}+B \boldsymbol{u}
        , \quad 
        \boldsymbol{x}(0)=\boldsymbol{x}_{0}\\
        &y=C x+D u
    \end{aligned}
    \end{equation}