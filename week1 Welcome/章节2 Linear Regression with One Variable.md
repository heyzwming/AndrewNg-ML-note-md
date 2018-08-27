二、Linear Regression with One Variable(单变量线性回归)
===

## Model and Cost Function

## 6、Model Representation(模型描述)

回归:将变量映射到某一个连续函数上，并预测实值输出。

![6.1](http://m.qpic.cn/psb?/V12umJF70r2BEK/CafuXmKosDXN.5FMGtIEq7n6ocrA1dXgyhPeYsX6wuI!/b/dDwBAAAAAAAA&bo=IQcDBAAAAAARBxE!&rf=viewer_4)

这章我们将这个问题简单地量化为**单变量线性回归模型**(Univariate linear regression)来理解它。

> To establish notation(建立符号) for future use, we’ll use $x^{(i)}$ to denote(表示) the **“input” variables** (living area in this example), also called **input features**,  
>
>  and $y^{(i)}$ to denote the **“output”** or **target variable** that we are trying to predict (price).  
>
>  A pair $(x^{(i)} , y^{(i)})$ is called **a training example**, and the dataset that we’ll be using to learn——a list of **m training examples** $(x^{(i)},y^{(i)})$; i=1,...,m—is called **a training set**. 
>
> Note that the superscript(上标) “$(i)$” in the notation is simply an **index** into the training set, and has nothing to do with exponentiation.
> 
>  We will also use **X** to denote the space of input values, and **Y** to denote the space of output values. In this example, $X$ = $Y$ = $ℝ$.

我们有一堆数据集，也叫训练集，下图我们来定义一些课程中用到的符号。

首先，我们定义三个变量：

m = 用于训练的样本数

$x^i$ = 第$i$个训练样本“输入”变量/特征量

$​y^i$ = ​第$i$个训练样本“输出”变量/特征量

$(x^{(i)} , y^{(i)})$ = 第$i$个训练示例

![6.2](http://m.qpic.cn/psb?/V12umJF70r2BEK/*ssrGbJhFGJCR0xMuxqlXZNyH.p.tXpTg3dWkqjX30o!/b/dIABAAAAAAAA&bo=NgcABAAAAAARFxU!&rf=viewer_4)


> To describe the supervised learning problem slightly more formally, our goal is, **given a training set**, to **learn a function** $h$ : $X → Y$ so that h(x) is a “good” predictor for the corresponding(一致的) value of $y$.
> 
>  For historical reasons, this function $h$ is called a hypothesis(假设). Seen pictorially, the process is therefore like this:
>
> When the target variable that we’re trying to predict is continuous, such as in our housing example, we call the learning problem a regression problem. When y can take on only a small number of discrete values (such as if, given the living area, we wanted to predict if a dwelling is a house or an apartment, say), we call it a classification problem.

如果我们要预测的目标变量是连续的，我们叫这类学习问题为回归问题，如果要预测的目标变量是连续的，我们叫这类学习问题为分类问题。

如何给训练集下定义，先来看一下监督学习算法是怎么工作的.

算法的任务是 输出一个函数，用小写字母$h$表示,$h$表示假设(hypothesis)函数,这个假设函数的作用是把房子的大小作为输入变量$x$值,并输出想应房子的预测$y$值。

接下来人们的问题变成了**如何表示假设函数**

假设函数$h$就是我们需要的这个拟合函数。

![6.3](http://m.qpic.cn/psb?/V12umJF70r2BEK/h0A6gdlaZzTGT3IvEPpZHyFSAihJUIvzfNCyPbnxvl8!/b/dIUBAAAAAAAA&bo=Swf8AwAAAAARF5M!&rf=viewer_4)


$$h_θ(x)=θ_0+θ_1*x$$            (1.1)
其中h是hypothesis（假设）的意思，当然，这个词在机器学习的当前情况下并不是特别准确。θ是参数，我们要做的是通过训练使得θ的表现效果更好。
这种模型被称为线性回归/单变量线性回归(Univariate linear regression)。


## 7、Cost Function(代价函数)

> We can measure the accuracy(精度) of our hypothesis function by using a cost function.
> 
>  This takes an average difference (actually a fancier version of an average) of all the results of the hypothesis with inputs from x's and the actual output y's.

本节中我们将定义代价函数的概念，这有助于我们弄清楚如何把最有可能的直线与我们的数据相拟合.这里的代价函数是通过计算假设函数预测的y值和实际值之间的差距，输出的y是假设函数$h$的精度/误差。



在这个假设函数 $h_\theta(x)=\theta_0+\theta_1x$中，$\theta_0$ 和 $\theta_1$ 我们把他们称为**模型参数**，我们要做的就是如何选择这两个参数值$\theta_0$ 和 $\theta_1$.

![7.1](http://m.qpic.cn/psb?/V12umJF70r2BEK/5FCPtEHttxyY9q5doh3MhFKYsLCsg*BuZY2dy4T1ftg!/b/dIUBAAAAAAAA&bo=CQfoAwAAAAARB9U!&rf=viewer_4)

不同的$\theta$有不同的假设和不同的假设函数。

![7.2](http://m.qpic.cn/psb?/V12umJF70r2BEK/icxayOXaOTc*FZZgoSudFYNi5FTs.M1ohkvRTMBhiG8!/b/dH4BAAAAAAAA&bo=Nwe9AwAAAAARF64!&rf=viewer_4)

我们现在有了数据集，并且可以通过改变参数来调整$h$函数，那么，我们如何定义什么是“更好”的$h$函数呢?

> 让我们给出标准的定义：在线性回归中，我们要解决的是一个最小化问题,所以我们要写出关于$\theta_0$和$\theta_1$的最小化，而且想要$h(x)$和y之间的差异尽可能小。即：通过调整$\theta$，使得所有训练集数据与其拟合数据的差的平方和更小，即认为得到了拟合度更好的函数。

我们引入了代价函数(平方误差函数/平方误差代价函数)：

$$ J(θ_0 ,θ_1)= \frac{1}{2m} \sum_{i=1}^{m}(h_\theta(x_i)-y_i)^2  $$


> This function is otherwise called the "Squared error function"(平方误差函数), or "Mean squared error"(均方误差). The mean is halved(均分) $(\frac{1}{2})$ as a convenience for the computation of the gradient descent(便于计算梯度下降), as the derivative term of the square function will cancel out the $\frac{1}{2}$ term( $\frac{1}{2}$会在求导的时候被消去 ). 

当代价函数$J$最小的时候(​minimize  $J(\theta_0, \theta_1)$)，即找到了对于当前训练集来说拟合度最高的函数h。

![7.3](http://m.qpic.cn/psb?/V12umJF70r2BEK/XbJqwpVJTFTQMKOLdprtzZVqbY7VUq.ovRVREXtUNx4!/b/dPQAAAAAAAAA&bo=LAcfBAAAAAARFxA!&rf=viewer_4)


## 8、Cost Function-Intuition Ⅰ(代价函数Ⅰ)

> If we try to think of it in visual terms(视觉层面), our training data set is scattered(散乱的) on the x-y plane. We are trying to make a straight line $h_\theta(x)$ which passes through these scattered data points.

回顾上节,并对假设函数$h$进行简化。

![8.1](http://m.qpic.cn/psb?/V12umJF70r2BEK/J9h77OYuMvj2nFjKYSL1X0WvOPqtmTmbNkSZhbu8Fgk!/b/dNoAAAAAAAAA&bo=Ege9AwAAAAARB5s!&rf=viewer_4)

Our objective is to get the best possible line. The best possible line will be such so that the average squared vertical distances of the scattered points from the line will be the least(散点与直线的平均垂直距离的平方最小。). Ideally(理想地), the line should pass through all the points of our training data set. In such a case, the value of $J(\theta_0, \theta_1)$ will be 0. The following example shows the ideal situation where we have a cost function of 0.



比较以下假设函数h和代价函数J
画出他们的图，以便更好地理解

When $\theta_1$ = 1 we get a slope of 1 which goes through every single data point in our model. 

当$\theta$为1时,代价函数$J(\theta) = 0$,左边假设函数$h$的曲线完美拟合数据集。

![8.2](http://m.qpic.cn/psb?/V12umJF70r2BEK/NGbgcTahmGGLFV5tLtchMhX4AOWGDCBRH5fKZxDX0VY!/b/dA0BAAAAAAAA&bo=WAcDBAAAAAARB2g!&rf=viewer_4)

Conversely, when $\theta_1$ = 0.5\, we see the vertical distance from our fit to the data points increase.This increases our cost function to 0.58.

当$\theta$为0.5时

![8.3](http://m.qpic.cn/psb?/V12umJF70r2BEK/ZIdWw4ioXH0wZB2YJlvZ.qV6GTphfo2Lo0x49bf058w!/b/dAsAAAAAAAAA&bo=TQf8AwAAAAARF5U!&rf=viewer_4)

 Plotting several other points yields to the following graph

当$\theta$为0时,最后可以预测到如下的代价函数$J(\theta)$的图像

![8.4](http://m.qpic.cn/psb?/V12umJF70r2BEK/DWzkwVQAeBdbMdW5WXBocWYXrOpf9K82FnEwRXVVnmA!/b/dNoAAAAAAAAA&bo=TAcVBAAAAAARF3o!&rf=viewer_4)
\
Thus as a goal, we should try to minimize the cost function. In this case, $\theta_1$ = 1 is our global minimum.

对于每一个θ，都可以得到一个不同的$J(\theta)$的值，也对应了一个不同的假设函数,对应左侧一条不同的直线和拟合程度。


## 9、Cost Function-Intuition Ⅱ(代价函数Ⅱ)

A contour plot(等值线图) is a graph that contains many contour lines. A contour line of a two variable function has a constant value(常数值) at all points of the same line. An example of such a graph is the one to the right below.

![9.1](http://m.qpic.cn/psb?/V12umJF70r2BEK/nLM.TKYWs0Ue9TQFUe4eOs9v*5iC.QYpX1B6qzs7QHc!/b/dN8AAAAAAAAA&bo=sgbhAgAAAAARB2c!&rf=viewer_4)

我们通过假设函数h和代价函数J来理解代价函数。

因为在本节中的代价函数J有两个变量θ_0和θ_1，所以在平面上无法得到J的图形。
![9.2](http://m.qpic.cn/psb?/V12umJF70r2BEK/q0VkyJ.2Yw8xEMQGhFr2A5rBzqy2IIIQMxgk139n1tQ!/b/dPQAAAAAAAAA&bo=VgcQBAAAAAARF2U!&rf=viewer_4)
![9.3](http://m.qpic.cn/psb?/V12umJF70r2BEK/WEG8jhan61YQTVb.czTMzHfAYBB2Vqp2qkWGXgR5FyQ!/b/dAsBAAAAAAAA&bo=aAbbAwAAAAARF5Y!&rf=viewer_4)

但是在下面我们会用等高线来展示这些曲面。

Taking any color and going along the 'circle', one would expect to get the same value of the cost function. 

其中的轴为θ_0和θ_1每一个椭圆展现了一系列$J(\Theta_0, \Theta_1)$值相等的点
对于我们研究的单变量线性回归而言，J函数关于θ的等高线图像大致如下：
![9.4](http://m.qpic.cn/psb?/V12umJF70r2BEK/eT4eAgl87HazwFobDNZZqR3foUYno8g9vLibsAITPN4!/b/dAsBAAAAAAAA&bo=RAcBBAAAAAARF2Y!&rf=viewer_4)

When $\theta_0$ = 360 and $\Theta_1$ = 0,the value of $J(\theta_0, \theta_1)$ in the contour plot gets closer to the center thus reducing the cost function error. Now giving our hypothesis function a slightly positive slope results in a better fit of the data.

![9.5](http://m.qpic.cn/psb?/V12umJF70r2BEK/OdVyC*TxXe4BaMvasyGjxnIBY6ElRSxRr6ecKRwgv6Q!/b/dOAAAAAAAAAA&bo=OAcQBAAAAAARFws!&rf=viewer_4)

The graph above minimizes the cost function as much as possible and consequently, the result of θ_1 and θ_0 tend to be around 0.12 and 250 respectively. Plotting those values on our graph to the right seems to put our point in the center of the inner most 'circle'.

![9.6](http://m.qpic.cn/psb?/V12umJF70r2BEK/*KaPz8aNu*8U2opDTNqUJo006NODfBB9bywtCk1OjAY!/b/dIUBAAAAAAAA&bo=7wZeAwAAAAARF5Q!&rf=viewer_4)

当我们找到了这些同心椭圆的中心点时，就找到了J函数的最小值，此时拟合度更好。

## Parameter Learning(参数学习)


## 10、Gradient Descent(梯度下降)

> So we have our hypothesis function and we have a way of measuring how well it fits into the data. Now we need to estimate(估计) the parameters in the hypothesis function. That's where gradient descent comes in.

> 梯度下降算法可以应用于更一般的($\theta_0 -> \theta_n$)，但为了简便符号，我们只使用 $\theta_0$ 和 $\theta_1$

梯度下降算法：我们先初始化$\theta_0$和 $\theta_1$为0,或者随意从某个($\theta_0$和 $\theta_1$)出发，然后不断尝试梯度地改变$\theta_0$和 $\theta_1$，来减小代价函数$J$的值，逐步逼近代价函数$J$的最小值。

![10.1](http://m.qpic.cn/psb?/V12umJF70r2BEK/eCDTH4rulqnrMeOCRiMehVPzskoUaGrOXO0u*M.kOjU!/b/dIABAAAAAAAA&bo=tAbAAwAAAAARB0E!&rf=viewer_4)

来看一个例子，假设我们随意初始化了一个值，我们站在这个图的某个高点，环顾四周，找到一条下降最快的路线，直到收敛至局部最低点。

> 梯度下降有个有趣的特点，第一次运行梯度下降法时，如果起点向右一点，梯度下降算法会得到一个完全不同的局部最优解。

The way we do this is by taking the derivative(求导) (the tangential line(切线) to a function) of our cost function. The slope(斜率)) of the tangent is the derivative at that point and it will give us a direction to move towards. We make steps down the cost function in the direction with the steepest(最陡的) descent. The size of each step is determined by the parameter $α$, which is called the learning rate.

![10.2](http://m.qpic.cn/psb?/V12umJF70r2BEK/vxOK6zUV4j*XGUh*fCsHLTuvoS9uvm*ldUrgrDMxr.I!/b/dA0BAAAAAAAA&bo=AAY8AwAAAAARFxk!&rf=viewer_4)
![10.3](http://m.qpic.cn/psb?/V12umJF70r2BEK/V4ou0V6gS6bi9D4abcUxbJY4r.zcmIx.YT4ZgbyzWIg!/b/dOAAAAAAAAAA&bo=9wUHAwAAAAARF9Y!&rf=viewer_4)

For example, the distance between each 'star' in the graph above represents a step determined by our parameter α. A smaller α would result in a smaller step and a larger α results in a larger step. The direction in which the step is taken is determined by the partial derivative of $J(\theta_0,\theta_1)$. Depending on where one starts on the graph, one could end up at different points. The image above shows us two different starting points that end up in two different places.

接下来看下算法的原理

repeat until convergence:

$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta_0, \theta_1)$$

where j=0,1 represents the feature index number.

At each iteration j, one should simultaneously update the parameters $\Theta_1$, $\Theta_2$,...,$\Theta_n$. Updating a specific parameter prior to calculating another one on the $j^{(th)}$ iteration would yield to a wrong implementation.


![算法原理10.4](http://m.qpic.cn/psb?/V12umJF70r2BEK/HVJGCpuZJIewxFJ4sjD0L4USLZhvQabMwj4L*.MvfhM!/b/dPQAAAAAAAAA&bo=TQcjBAAAAAARF00!&rf=viewer_4)

$:=$ 表示赋值

$α$ 表示 learning rate 即梯度下降的速率\多大的幅度更新参数$\theta_j$

实现 梯度下降算法的微妙之处 是 ,对于这个表达式(更新方程)，你需要同时更新(simultaneously update)$\theta_0$ 和 $\theta_1$,即$\theta_0$更新为$\theta_0$减去某项，$\theta_1$同理,而不是像Incorrect中的这样，因为会改变代价函数J的值，导致生成的temp1出错。

## 11、Gradient Descent Intuition(梯度下降的直觉)


先看下上节课的这个更新表达式

Repeat until convergence:

$$\theta_1:=\theta_1-\alpha \frac{d}{d\theta_1} J(\theta_1)$$

Regardless of the slope's sign for $\frac{d}{d\theta_1} J(\theta_1)$, $\theta_1$ eventually converges to its minimum value. 


![11.1更新表达式](http://m.qpic.cn/psb?/V12umJF70r2BEK/z5vuB1.2jpp32YLo9E1ODUSJWZ6M7yRZKNG1YeovF38!/b/dA0BAAAAAAAA&bo=KAYDAwAAAAARBx4!&rf=viewer_4)

下面解释下导数项的意义

The following graph shows that when the slope is negative, the value of $\theta_1$ increases and when it is positive, the value of $\theta_1$ decreases.

当$\theta$大于最小值时，导数为正，那么迭代公式里，$\theta$减去一个正数，向左往最小值逼近；

当$\theta$小于最小值时，导数为负，那么迭代公式​里，$\theta$减去一个负数，向右往最小值逼近；

![11.2导数项的意义](http://m.qpic.cn/psb?/V12umJF70r2BEK/W4sZ0OBiKFOmqN2o5hyahVp6AwFmGDoebk56oUgzFLI!/b/dNoAAAAAAAAA&bo=7wYNBAAAAAARF8A!&rf=viewer_4)

学习速率$α$的作用:
如果$α$太小，梯度下降的速度可能很慢
$α$太大则会一次次越过最低点,它会导致无法收敛甚至发散。

![11.3α的作用](http://m.qpic.cn/psb?/V12umJF70r2BEK/uHuIE1qRJEJsHdxqaoKtAFY6IqJsFOe8BCJeLcPe3yg!/b/dN4AAAAAAAAA&bo=.AbuAwAAAAARFzM!&rf=viewer_4)

How does gradient descent converge with a fixed step size $\alpha$?

The intuition(直觉) behind the convergence(收敛) is that $\frac{d}{d\theta_1}$$J(\theta_1)$ approaches 0 as we approach the bottom of our convex function. At the minimum, the derivative will always be 0 and thus we get:

$\theta_1:=\theta_1-\alpha * 0$


如果$\theta_1$已经处在一个局部最优点，下一步梯度下降会怎样？ 
显然，$\theta_1$不再改变。

![11.4](http://m.qpic.cn/psb?/V12umJF70r2BEK/Af3DFL6qej6WcZO0Bce.hP0FsKIH.tNQdyxzeIJzR0w!/b/dN0AAAAAAAAA&bo=nwX0AgAAAAARF0w!&rf=viewer_4)

当我们接近局部最低时，导数值会自动变得越来越小,所以梯度下降将自动采取较小的幅度,这就是梯度下降的运行方式。

![11.5](http://m.qpic.cn/psb?/V12umJF70r2BEK/JiZJr.gdHJZJHjqVgvM*kVNNXiYDfKo8n3.l.u98iYA!/b/dPQAAAAAAAAA&bo=cQUQAwAAAAARF0c!&rf=viewer_4)

## 12、Gradient Descent For Linear Regression(线性回归的梯度下降法)

**Note**: [At 6:15 "h(x) = -900 - 0.1x" should be "h(x) = 900 - 0.1x"]

When specifically applied to the case of linear regression, a new form of the gradient descent equation can be derived. We can substitute our actual cost function and our actual hypothesis function and modify the equation to :

repeat until convergence: {
    
$$\theta_0 := \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^m ( h_\theta(x_i)-y_i ) $$

$$\theta_1 := \theta_1 - \alpha \frac{1}{m} \sum_{i=1}^m (( h_\theta(x_i)-y_i ) x_i) $$
}



where m is the size of the training set, $\theta_0$ a constant that will be changing simultaneously with $\theta_1$ and $x_{i}, y_{i}$ are values of the given training set (data).

Note that we have separated out the two cases for $\theta_j$ into separate equations for $\theta_0$ and $\theta_1$ ; and that for $\theta_1$ we are multiplying $x_{i}$ at the end due to the derivative. The following is a derivation of $\frac {\partial}{\partial \theta_j}$$J(\theta)$ for a single example :


The point of all this is that if we start with a guess for our hypothesis and then repeatedly apply these gradient descent equations, our hypothesis will become more and more accurate.

So, this is simply gradient descent on the original cost function J. This method looks at every example in the entire training set on every step, and is called **batch gradient descent**. Note that, while gradient descent can be susceptible to local minima in general, the optimization problem we have posed here for linear regression has only one global, and no other local, optima; thus gradient descent always converges (assuming the learning rate α is not too large) to the global minimum. Indeed, J is a convex quadratic function. Here is an example of gradient descent as it is run to minimize a quadratic function.


The ellipses shown above are the contours of a quadratic function. Also shown is the trajectory taken by gradient descent, which was initialized at (48,30). The x’s in the figure (joined by straight lines) mark the successive values of θ that gradient descent went through as it converged to its minimum.


本节，我们要将梯度下降和代价函数结合,得到线性回归的算法,它可以用直线模型来拟合数据。
下图是 梯度下降法 和 线性回归模型(包括线性假设和平方差代价函数)
![12.1](http://m.qpic.cn/psb?/V12umJF70r2BEK/9jMFxpwtfFuBPv7i*11Vhyd0rgU8o7zMUu4uJZ*XNhA!/b/dOAAAAAAAAAA&bo=lwXqAgAAAAARB0o!&rf=viewer_4)
求解代价函数J中的偏导项
![12.2](http://m.qpic.cn/psb?/V12umJF70r2BEK/XVWfx6Elw4VOQnlg64E3KliQFN.FW*2AApMiDDXm87Q!/b/dPQAAAAAAAAA&bo=sQXBAgAAAAARF1c!&rf=viewer_4)

把他们代回梯度下降算法，这里是回归的梯度下降法
![12.3](http://m.qpic.cn/psb?/V12umJF70r2BEK/Xe0lnQs638Q5k.KVnEcnH1HuUBVHuVVMKHyiIRBB3YA!/b/dAsBAAAAAAAA&bo=egW0AgAAAAARF.k!&rf=viewer_4)

线性回归的代价函数总是像一个碗装的弓状函数,术语叫做凸函数
## 13、Review
