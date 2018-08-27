## 六、Logistic Regression(Logistic回归)
===

### Classification and Representation

28、Classification

29、Hypothesis Representation

30、Decision Boundary

### Logistic Regression Model

31、Cost Function

32、Simplified Cost Function and Gradient Descent

33、Advanced Optimization

### Multiclass Classification

34、Multiclass Classification:One-vs-all

### Review
 


# Classification and Representation

## 课时42  分类(Classification)  08:08
 
预测值是离散值情况下的分类问题
我们从预测值只有0、1两类的分类问题入手。

![42.1](http://m.qpic.cn/psb?/V12umJF70r2BEK/DxzpST2fKpFyUpkmxHHkVAhrjl7YpoC8yzUSfA7escw!/b/dPQAAAAAAAAA&bo=CwRIAgAAAAARB3U!&rf=viewer_4)

在分类问题中应用线性回归不是一个好主意。因为一些额外的样本可能会影响线性回归函数的数据拟合度。

![42.2](http://m.qpic.cn/psb?/V12umJF70r2BEK/xnpNGC0VSqldkNY0Ucw0ZKZDgjyOaG84oZ6wiRliUik!/b/dIUBAAAAAAAA&bo=NgRuAgAAAAARF34!&rf=viewer_4)

对于分类问题，y的值是离散的$0$或1，如果使用线性回归，假设的输出值会远大于1或者小于$0$，即使所有的训练样本的标签都是y=0或1

我们要使用一种新的算法叫做Logistac Regression(应用于分类问题)，特点在于算法的输出或者说预测值一直介于0和1之间

## 课时43  假设陈述(Hypothesis Representation)  07:24
logistic回归 中假设函数的表示方法

我们希望我们的假设函数$h_\theta(x)$的输出能够在0到1之间。

当我们使用线性回归的时候，假设函数是这样的$h_\theta(x) = \theta^Tx$.其中$\theta$是参数的向量，x是特征值的向量，如下。

$$
h_\theta(x) = \left[  \theta_0  .  \theta_1 \dots \theta_n  \right]
 \left[
 \begin{matrix}
   x_0  \\
   x_1  \\
   ·\\
   ·\\
   · \\
   x_n
  \end{matrix}
  \right] = \theta^T x$$

我们定义个函数 $g(z)=\frac{1}{1+e^{-z}}$ ,而$z = \theta^Tx$,这个函数$g$就叫Sigmoid函数或者Logistic函数，把$z$代入$h_\theta(x)$得到

$$h_\theta(x)=\frac{1}{1+e^{-{\theta^Tx}}}$$

我们要做的就是用参数θ拟合我们的数据，拿到一个训练集，我们需要给参数$θ$选定一个值，假设函数会帮我们做出预测。

![43.1](http://m.qpic.cn/psb?/V12umJF70r2BEK/Y42BroZryYy.ZS1xFOsi23CI.rYVcdZQ9v48F*QKrKU!/b/dPQAAAAAAAAA&bo=RwRsAgAAAAARBx0!&rf=viewer_4)


先解释下这个函数$h$输出的含义  
当函数$h$输出一个数字，我把这个数字当成 对一个输入x，y=1的概率估计。

比方说$h_θ(x) = 7$的意义是 (在单特征分类问题下)对于一个特征值为x的患者y=1的概率是0.7，用数学来表示，$h_θ(x)$是 在特征值为x(此处是肿瘤的大小)和参数θ的条件下，y=1的概率p。(第0个特征是1，第1个特征是肿瘤大小)

假设函数$h_\theta(x)$的数学表达式是$P(y=1|x;\theta)$

![43.2](http://m.qpic.cn/psb?/V12umJF70r2BEK/LBlNySrvssa1rM9aUzPi182*dfMdxa4ETPIXry6fEj0!/b/dN8AAAAAAAAA&bo=NgRxAgAAAAARB3E!&rf=viewer_4)


## 课时44  决策界限(Decision Boundary)  14:49

决策边界的概念，能帮助我们更好的理解logistic回归的假设函数在计算什么。
事实上这个假设函数计算的是 在特征x和参数θ的条件下y=1的估计概率。

如果这个概率>=0.5 我们预测y = 1
否则 y = 0

看到这个图像我们可以看到如果z >= 0 那么g(z) >= 0.5
即当$θ^Tx >= 0$.我们的假设函数就会预测y = 1

![44.1](http://m.qpic.cn/psb?/V12umJF70r2BEK/3OX1YKgf9ptwMk08udNS6CluSzOiByEQg73oUtXy4wM!/b/dD0BAAAAAAAA&bo=IQRgAgAAAAARF2c!&rf=viewer_4)

通过这些我们可以更好的理解Logistic回归的假设函数是如何做出预测的。

我们假设我们的假设函数是$h_\theta(x) = g(\theta_0+\theta_1x_1+\theta_2x_2)$

为了拟合下图中的数据集，我们假设已经拟合好了参数$\theta$并有这么一个参数向量$θ = \begin{bmatrix}
-3 \\ 1 \\ 1
\end{bmatrix}$ 来更深入理解假设函数何时为1何时为0

当$\theta^Tx=-3+x_1+x_2>=0$时预测y=1

$x_1+x_3 = 3$作出的区别开y = 1 和y = 0范围的线就叫决策边界 

![44.2](http://m.qpic.cn/psb?/V12umJF70r2BEK/Jfu1Kz5VNOpDiVTf7wC5EX9H1xwM63mS2WGxSIpPK6I!/b/dGwBAAAAAAAA&bo=PARyAgAAAAARF2g!&rf=viewer_4)


接下来是一个更加复杂的例子
我们添加额外的特征量$x_1^2$ 和$x_2^2$
我们选择参数向量$θ=\begin{bmatrix}
    -1 & 0 & 0 & 1 & 1
\end{bmatrix}$
通过更复杂的多项式，我们可以得到更复杂的决定边界，决定边界不是训练集的属性，而是假设本身及其参数的属性，只要给定了参数向量θ，圆形的决定边界就确定了。  
我们用训练集来拟合参数θ

![44.3](http://m.qpic.cn/psb?/V12umJF70r2BEK/YTJXADnKrdjFC.FnxQTrT8bBmYOAivJcCJubimHT8FA!/b/dIUBAAAAAAAA&bo=KwRWAgAAAAARF1s!&rf=viewer_4)

最后再来看下更复杂 更高阶多项式的情况，这会让你得到非常复杂的决策边界。

![44.4](http://m.qpic.cn/psb?/V12umJF70r2BEK/2gucMaUnUE.YriIA3f3sCs*NOCrjplzruwvFuaQVKMI!/b/dN4AAAAAAAAA&bo=JgRMAgAAAAARB1w!&rf=viewer_4)



# Logistic Regression Model

## 课时45  代价函数(Cost Function)  10:23

如何拟合logistic回归模型的参数θ

定义用来拟合参数的优化目标或者叫代价函数。
这就是logistic回归模型的拟合问题。

我们有一个训练集，每一个样本都用n+1维的特征向量表示。

对于给定的训练集我们如何选择 或者说如何拟合 参数θ
![45.1](http://m.qpic.cn/psb?/V12umJF70r2BEK/oAxytbF8DMNSaYT7c3qo0XlMcrYMPpk3pTBwCnnkJq0!/b/dGwBAAAAAAAA&bo=LwQ8AgAAAAARByU!&rf=viewer_4)

我们定义$Cost(h_\theta(x),y)$，这个代价函数的理解是这样的，它是在输出的预测值是h(x)时而实际标签是y的情况下，我们希望学习算法付出的代价。  
但是如果我们使用这个Cost代价函数，它会变成参数$\theta$的非凸函数.  
如何我们把$h_\theta(x) = \frac{1}{1+e^{\theta^Tx}}$代入$Cost(h_\theta(x),y)$中，再把$Cost$带入$J(\theta)$,会得到关于参数$\theta$的非凸函数。

![45.2](http://m.qpic.cn/psb?/V12umJF70r2BEK/kwJ99dZKnG6ClPiMqFeoUFadC4hXBtqJksPxu4J7EWo!/b/dA0BAAAAAAAA&bo=HgRPAgAAAAARF3c!&rf=viewer_4)

所以我们要找个别的代价函数

$$Cost(h_\theta(x),y) = \begin{cases}
-log{(h_\theta(x))} & y = 1 \\
-log(1-h_\theta(x)) & y = 0
\end{cases}$$

这个代价函数有一些有趣而且很好的性质  
* 如果假设函数的预测值是1，而且y刚好等于我的预测，那它的代价是0
* 如果预测值是0而y = 1则代价是$\infty$

![45.3](http://m.qpic.cn/psb?/V12umJF70r2BEK/k.6D7TGkLbKOY917IFiqtly95iOl7JwwV3rl36ftvc8!/b/dGwBAAAAAAAA&bo=MQRiAgAAAAARF3U!&rf=viewer_4)

当y=0时同理

![45.4](http://m.qpic.cn/psb?/V12umJF70r2BEK/yo0XMDT5QZiT3sk3uT8cp5OzTYROdiGnr8be*kBSLMQ!/b/dN4AAAAAAAAA&bo=HARVAgAAAAARF28!&rf=viewer_4)


## 课时46  简化代价函数与梯度下降(Simplified Cost Function and Gradient Descent)    10:14
简化代价函数(将y = 0和y = 1两种情况合并)和如何用梯度下降法拟合出logistic回归的参数

#### Logistic regression cost function
$$J(\theta)=\frac{1}{m} \sum_{i = 1}^mCost(h_\theta(x^{(i)}),y^{(i)})$$

所有样本的代价和

$$Cost(h_\theta(x),y) = \begin{cases}
-log(h_\theta(x)) & y = 1 \\
-log(1-h_\theta(x)) & y = 0
\end{cases}$$

单个样本的代价

$Note$: $y$ = 0 or 1 always

将代价函数Cost简化：  
$$Cost(h_\theta(x^{(i)}),y^{(i)})= -y·log(h_\theta(x))-(1-y)·log(1-h_\theta(x))$$

通过将$y=1$或者$y=0$代入可证上式。

最后我们得到
$$J(\theta) = \frac{1}{m}\sum_{i=1}^m Cost(h_\theta(x^{(i)}),y^{(i)}) $$
$$= -\frac{1}{m}[\sum_{i=1}^m  y^{(i)}·log(h_\theta(x^{(i)}))+(1-y^{(i)})·log(1-h_\theta(x^{(i)}))]$$

对这个代价函数，我们要拟合出参数$\theta$要做的是找出让$J(\theta)$取得最小值的$\theta$


![46.1](http://m.qpic.cn/psb?/V12umJF70r2BEK/GoygYiHqKmuY4fkmRhfK2.3uBJvJovLMI6*WSU.hIes!/b/dN8AAAAAAAAA&bo=HgRFAgAAAAARF30!&rf=viewer_4)

现在我们要获得最小化关于$θ$的代价函数$J(\theta)$的方法.
如果你有n个特征，你就有一个参数向量$θ = \begin{bmatrix}
\theta_0 \\ \theta_1 \\ \theta_2 \\ \vdots \\ \theta_n
\end{bmatrix}$

线性回归和Logistic回归的梯度下降算法的更新规则公式长得一模一样，但是其中的意义因为假设函数$h_\theta(x)$不同而不同

![46.2](http://m.qpic.cn/psb?/V12umJF70r2BEK/XN4eIPz3gyMIfvsOqKgqNNu5Yz1rlwOzio1Tb0Y5iL4!/b/dA0BAAAAAAAA&bo=HQRIAgAAAAARF3M!&rf=viewer_4)

另外，特征缩放的方法也可以让Logistic回归算法梯度下降收敛更快

## 课时47  高级优化(Advanced Optimization)  14:06




# Multiclass Classification

## 课时48  多元分类：一对多(Multiclass Classification:One vs all)  06:15



## 课时49  本章课程总结