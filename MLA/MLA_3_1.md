# Assignment 3

## 1 Mathematics

Nonlinear least-squares. Suppose that $f(x):\mathbb{R}^n \rightarrow \mathbb{R}^n, x \in \mathbb{R}^n, f \in \mathbb{R}^n$, and some $f_i(x):\mathbb{R}^n \rightarrow \mathbb{R}$is a (are) non-linear function(s). Then, the problem,

$$\vec{x}^*=\underset{x}{arg\,min} \frac{1}{2} ||f(\vec{x})||_2^2 = \underset{x}{arg\,min} \frac{1}{2}(f(\vec{x}))^T f(\vec{x})$$
is a nonlinear least-squares problem. In our lecture, we mentioned that Levenberg-Marquardt algorithm is a typical method to solve this problem. In L-M algorithm, for each updating step, at the current $\vec{x}$, a local approximation model is constructed as,
$$L(\boldsymbol{h})=\frac{1}{2} (f(\boldsymbol{x}+\boldsymbol{h}))^T f(\boldsymbol{x}+\boldsymbol{h}) + \frac{1}{2} \mu \boldsymbol{h}^T\boldsymbol{h} \\
=\frac{1}{2}(f(\boldsymbol{x}))^T f(\boldsymbol{x}) + \boldsymbol{h}^T(J(\boldsymbol{x}))^T f +\frac{1}{2}\boldsymbol{h}^T(J(\boldsymbol{x}))^T J(\boldsymbol{x}) \boldsymbol{h} + \frac{1}{2}\mu \boldsymbol{h}^T\boldsymbol{h}$$
where $\boldsymbol{J}(\boldsymbol{x})$ is $\boldsymbol{f}(\boldsymbol{x})$'s Jacobian matrix, and $\mu >0$ is the damped coefficient. Please prove that $L(\boldsymbol{h})$ is a strictly convex function. (Hint: If a function $L(\boldsymbol{x})$ is differentiable up to at least second order, $\boldsymbol{L}$ is stricly convex if its Hessian matrix is positive definite.)

Proof：
$$
\begin{aligned}
    \because & L(\boldsymbol{h})=\frac{1}{2}(\boldsymbol{f}(\boldsymbol{x}))^T \boldsymbol{f}(\boldsymbol{x}) + \boldsymbol{h}^T(\boldsymbol{J}(\boldsymbol{x}))^T\boldsymbol{f} + \frac{1}{2}\boldsymbol{h}^{T}(\boldsymbol{J}(\boldsymbol{x}))^T(\boldsymbol{J}(\boldsymbol{x}))\boldsymbol{h} + \frac{1}{2}\mu \boldsymbol{h}^T \boldsymbol{h} \\

    & \boldsymbol{A} \in \mathbb{R^{n\times n}}, \vec{x} \in\mathbb{R^{n\times 1}}, then \frac{d\vec{x}^T\boldsymbol{A}\vec{x}}{d\vec{x}}=(\boldsymbol{A} + \boldsymbol{A}^T)\vec{x}\\
    & \boldsymbol{A} \in \mathbb{R^{m\times n}}, \vec{x} \in\mathbb{R^{n\times 1}}, then \frac{d\vec{x}^T\boldsymbol{A}^T}{d\vec{x}}=\boldsymbol{A}^T \\
    & \boldsymbol{f}(\boldsymbol{x}) \text{ is independent of }\boldsymbol{h} \\
    \therefore &\frac{d\boldsymbol{L}}{d\boldsymbol{h}} = \frac{d\boldsymbol{h}^T}{d\boldsymbol{h}}\boldsymbol{J}^T\boldsymbol{f} + \frac{1}{2}(\boldsymbol{J}^T\boldsymbol{J}+(\boldsymbol{J}^T\boldsymbol{J})^T)\boldsymbol{h} + \mu \boldsymbol{h}\\
    &=\boldsymbol{J}^T\boldsymbol{f} +\frac{1}{2}(\boldsymbol{J}^T\boldsymbol{J}+(\boldsymbol{J}^T\boldsymbol{J})^T)\boldsymbol{h} + \mu \boldsymbol{h}\\
    &= \boldsymbol{J}^T\boldsymbol{f} +\boldsymbol{J}^T\boldsymbol{J}\boldsymbol{h} + \mu \boldsymbol{h}\\
    &=\boldsymbol{J}^T\boldsymbol{f} +(\boldsymbol{J}^T\boldsymbol{J} + \mu \boldsymbol{E})\boldsymbol{h} \\
    & \frac{d^2\boldsymbol{L}}{d\boldsymbol{h}^2}= \boldsymbol{J}^T\boldsymbol{J} + \mu \boldsymbol{E}\\
    \because &\boldsymbol{J}^T\boldsymbol{J} \text{is positive semi-definite.} \\
    \therefore &\boldsymbol{J}^T\boldsymbol{J}\text{'s eigen-value}\geq0\\
    \because &\mu >0\\
    \therefore  &(\boldsymbol{J}^T\boldsymbol{J} + \mu \boldsymbol{E})\text{'s eigen-value}>0\\
    \therefore &(\boldsymbol{J}^T\boldsymbol{J} + \mu \boldsymbol{E})\text{is positive definite matrix.}\\
    \therefore &\boldsymbol{H}(\boldsymbol{L})>0\\
    \Rightarrow & \boldsymbol{L}(\boldsymbol{h})\text{is a strictly convex function.}
\end{aligned}
$$

<!-- & \text{Let } \boldsymbol{L(\boldsymbol{h})} = \boldsymbol{0}\\
    & \boldsymbol{h}=-(\boldsymbol{J}^T\boldsymbol{J} + \mu \boldsymbol{E})^{-1} \boldsymbol{J}^T\boldsymbol{f}\\ -->

## 2 Programming

A matlab program demonstrating how to estimate the parameters of a GMM using the EM algorithm.

```matlab
clear;
close all;

datapath = 'data3.mat';
load(datapath); 

figure;
h1 = gscatter(data(:,1), data(:,2)); 
hold on;

%请给下面这个section加上总的注释，并尽量给每一个语句加上注释
xRange = min(data(:,1)) :0.1:max(data(:,1));
yRange = min(data(:,2)):0.1:max(data(:,2));
xRange = xRange * 1.5; 
yRange = yRange * 1.5;
[gridX, gridY] = meshgrid(xRange, yRange); 
%%%%%%%%%%%%%%


set(gca, 'color', [0.3 0.5 0.6])
set(gcf, 'color', [0.3 0.5 0.6])

% create a gif from the plotting option
gif = true;
if gif
    filename = 'gmm.gif';
    delete(filename);
end

%% 请加上注释来说明一下这几个变量代表或存储的都是什么东西？
K = 3; 
pDatatoEachGauss = zeros(size(data, 1), K); 
mus = zeros(K, size(data, 2)); 
alphas = ones(K, 1) / K; 
sigmas = struct(); 
%%%%%%%%%%%%%%%%%%%


for indexGauss = 1:K
    %%%%%%%%%%%%%%%%
    %给这个section里面的每一句话加上注释
    sample = data(randsample(1:size(data,1), 1), :); 
    mus(indexGauss, :) = sample; 
    sigmas(indexGauss).covmat = 0.1 * rand(1,1) * cov(data); 
    
    gaussValuesAtSamplingPoints = mvnpdf([gridX(:) gridY(:)], mus(indexGauss, :), sigmas(indexGauss).covmat); 
    gaussValuesAtSamplingPoints = reshape(gaussValuesAtSamplingPoints, length(yRange), length(xRange));
    [~, hn(indexGauss)] = contour(xRange, yRange, gaussValuesAtSamplingPoints, [.0001 .001 .01 .05:.1:.95 .99 .999 .9999]); 
    hmus(indexGauss) = plot(mus(indexGauss,1), mus(indexGauss,2), 'kx', 'LineWidth', 2, 'MarkerSize', 10);
    %%%%%%%%%%%%%%%%%%%%%%%
    
    frame = getframe(1);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
end

%% EM iteration steps
epsilon = 0.001;
iteration = 1;
loglikelihood = [];
while(true)
    % 对这个for循环加上整体的注释
    for dataIndex = 1 : size(data,1)
        pThisDataToEachGauss = arrayfun(@(k) alphas(k) * mvnpdf(data(dataIndex, :), mus(k, :), sigmas(k).covmat), 1:K);
        pThisDataToEachGauss = pThisDataToEachGauss / sum(pThisDataToEachGauss);
        pDatatoEachGauss(dataIndex, :) = pThisDataToEachGauss;
    end
    
    %为本句加上注释
    sumPDataToEachGauss = sum(pDatatoEachGauss, 1);
    %为本句加上注释
    alphas = (sumPDataToEachGauss/sum(sumPDataToEachGauss))';
    
    for gaussIndex = 1 : K
        %%%%%%%%%%%%%%%%%%%%
        %为这个section里的每个语句加上注释
        gaussMuUpper = sum(data(:, :) .* pDatatoEachGauss(:, gaussIndex));
        mus(gaussIndex,:) = gaussMuUpper / sumPDataToEachGauss(gaussIndex);
        sigmas(gaussIndex).covmat = ((data(:,:)-mus(gaussIndex, :))' * (pDatatoEachGauss(:, gaussIndex) .* (data(:,:)-mus(gaussIndex, :))))/sumPDataToEachGauss(gaussIndex);
        
        delete(hn(gaussIndex));
        delete(h1);
        delete(hmus(gaussIndex));
        
        [M,I] = max(pDatatoEachGauss, [], 2);
        h1 = gscatter(data(:,1), data(:,2), I);
        
        F = mvnpdf([gridX(:) gridY(:)], mus(gaussIndex, :), sigmas(gaussIndex).covmat);
        F = reshape(F, length(yRange), length(xRange));
        [~, hn(gaussIndex)] = contour(xRange, yRange, F, [.0001 .001 .01 .05:.1:.95 .99 .999 .9999]); 
        hmus(gaussIndex) = plot(mus(gaussIndex,1),mus(gaussIndex,2),'kx','LineWidth',2,'MarkerSize',10);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        uistack(h1, 'bottom');
        drawnow;
        
        frame = getframe(1);
        im = frame2im(frame);
        [imind,cm] = rgb2ind(im,256);
        imwrite(imind,cm,filename,'gif','WriteMode','append');
    end
    
    %%%%%%%%%%%%%%%%%%
    %给这个section里面的每个语句加上注释
    laccum = 0; 
    for dataIndex = 1 : size(data, 1)
        thisDataPointLiklyhood = 0;
        for gaussIndex = 1 : K
            thisDataPointLiklyhood = thisDataPointLiklyhood + alphas(gaussIndex)* mvnpdf(data(dataIndex,:), mus(gaussIndex, :), sigmas(gaussIndex).covmat);
        end
        laccum = laccum + log(thisDataPointLiklyhood);
    end
    %%%%%%%%%%%%%%%%%%
    fprintf('[%d-th iteration] log-likelihood: %.3f\n', iteration, laccum);
    loglikelihood = [loglikelihood; laccum];
    iteration = iteration + 1;
    
    % 迭代停止条件：如果两次迭代的对数似然之差小于eps了，迭代停止
    if numel(loglikelihood) > 1
        if abs(loglikelihood(end)-loglikelihood(end-1)) <= epsilon
            fprintf('[Optimization completed]\n');
            break;
        end
    end
end
```