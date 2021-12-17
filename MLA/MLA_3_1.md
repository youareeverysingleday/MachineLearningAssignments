# 1

Nonlinear least-squares. Suppose that $f(x):\mathbb{R}^n \rightarrow \mathbb{R}^n, x \in \mathbb{R}^n, f \in \mathbb{R}^n$, and some $f_i(x):\mathbb{R}^n \rightarrow \mathbb{R}$is a (are) non-linear function(s). Then, the problem,

$$\vec{x}^*=\underset{x}{arg\,min} \frac{1}{2} ||f(\vec{x})||_2^2 = \underset{x}{arg\,min} \frac{1}{2}(f(\vec{x}))^T f(\vec{x})$$
is a nonlinear least-squares problem. In our lecture, we mentioned that Levenberg-Marquardt algorithm is a typical method to solve this problem. In L-M algorithm, for each updating step, at the current $\vec{x}$, a local approximation model is constructed as,
$$L(\boldsymbol{h})=\frac{1}{2} (f(\boldsymbol{x}+\boldsymbol{h}))^T f(\boldsymbol{x}+\boldsymbol{h}) + \frac{1}{2} \mu \boldsymbol{h}^T\boldsymbol{h} \\
=\frac{1}{2}(f(\boldsymbol{x}))^T f(\boldsymbol{x}) + \boldsymbol{h}^T(J(\boldsymbol{x}))^T f +\frac{1}{2}\boldsymbol{h}^T(J(\boldsymbol{x}))^T J(\boldsymbol{x}) \boldsymbol{h} + \frac{1}{2}\mu \boldsymbol{h}^T\boldsymbol{h}$$
where $\boldsymbol{J}(\boldsymbol{x})$ is $\boldsymbol{f}(\boldsymbol{x})$'s Jacobian matrix, and $\mu >0$ is the damped coefficient. Please prove that $L(\boldsymbol{h})$ is a strictly convex function. (Hint: If a function $L(\boldsymbol{x})$ is differentiable up to at least second order, $\boldsymbol{L}$ is stricly convex if its Hessian matrix is positive definite.)