Logistic 回归(Logistic Regression)
===



## 七、Regularization
---
### Solving the Problem of Overfitting

35、The Problem of Overfitting

36、Cost Function

37、Regularized Linear Regression

38、Regularized Logistic Regression

### Review






# Classification and Representation

## 课时42  分类(Classification)  08:08
 
预测值是离散值情况下的分类问题
我们从只有01两类的分类问题入手。

![42.1](http://m.qpic.cn/psb?/V12umJF70r2BEK/DxzpST2fKpFyUpkmxHHkVAhrjl7YpoC8yzUSfA7escw!/b/dPQAAAAAAAAA&bo=CwRIAgAAAAARB3U!&rf=viewer_4)

在分类问题中应用线性回归不是一个好主意。因为一些额外的样本可能会影响线性回归函数的数据拟合度。

![42.2](http://m.qpic.cn/psb?/V12umJF70r2BEK/xnpNGC0VSqldkNY0Ucw0ZKZDgjyOaG84oZ6wiRliUik!/b/dIUBAAAAAAAA&bo=NgRuAgAAAAARF34!&rf=viewer_4)

对于分类问题，y的值是离散的0或1，如果使用线性回归，假设的输出值会远大于1或者小于0，即使所有的训练样本的标签都是y=0或1

我们要使用一种新的算法叫做Logistac Regression(应用于分类问题)，特点在于算法的输出或者说预测值一直介于0和1之间

## 课时43  假设陈述(Hypothesis Representation)  07:24
logistic回归 中假设函数的表示方法

我们要做的就是用参数θ拟合我们的数据，拿到一个训练集，我们需要给参数θ选定一个值，假设会帮我们做出预测。

![43.1](http://m.qpic.cn/psb?/V12umJF70r2BEK/Y42BroZryYy.ZS1xFOsi23CI.rYVcdZQ9v48F*QKrKU!/b/dPQAAAAAAAAA&bo=RwRsAgAAAAARBx0!&rf=viewer_4)

h_θ(x) = 7，的意义是 对于一个特征为x的患者y=1的概率是0.7，用数学来表示，h_θ(x)是 在特征值x(此处是肿瘤的大小)和参数θ的条件下，y=1的概率p。
假设函数表达式
定义逻辑回归的假设函数的数学公式

![43.2](http://m.qpic.cn/psb?/V12umJF70r2BEK/LBlNySrvssa1rM9aUzPi182*dfMdxa4ETPIXry6fEj0!/b/dN8AAAAAAAAA&bo=NgRxAgAAAAARB3E!&rf=viewer_4)


## 课时44  决策界限(Decision Boundary)  14:49

决策边界的概念，能帮助我们更好的理解logistic回归的假设函数在计算什么。
事实上这个假设函数计算的是 在特征x和参数θ的条件下y=1的估计概率。

如果这个概率>=0.5 我们预测y = 1
否则 y = 0

看到这个图像我们可以看到如果z >= 0 那么g(z) >= 0.5
即当θ^T*x >= 0.我们的假设函数就会预测y = 1

![44.1](http://m.qpic.cn/psb?/V12umJF70r2BEK/3OX1YKgf9ptwMk08udNS6CluSzOiByEQg73oUtXy4wM!/b/dD0BAAAAAAAA&bo=IQRgAgAAAAARF2c!&rf=viewer_4)

为了拟合下图中的数据集，我们假设有这么一个参数向量θ = [-3 1 1]^T 来更深入理解假设函数何时为1何时为0

区别开y = 1 和y = 0范围的线就叫决策边界 

![44.2](http://m.qpic.cn/psb?/V12umJF70r2BEK/Jfu1Kz5VNOpDiVTf7wC5EX9H1xwM63mS2WGxSIpPK6I!/b/dGwBAAAAAAAA&bo=PARyAgAAAAARF2g!&rf=viewer_4)

接下来是一个更加复杂的例子
我们添加额外的特征量x1^2 和x2^2
我们选择参数向量θ为[-1 0 0 1 1]
通过更复杂的多项式，我们可以得到更复杂的决定边界，决定边界不是训练集的属性，而是假设本身及其参数的属性，只要给定了参数向量θ，圆形的决定边界就确定了。
我们用训练集来拟合参数θ

![44.3](http://m.qpic.cn/psb?/V12umJF70r2BEK/YTJXADnKrdjFC.FnxQTrT8bBmYOAivJcCJubimHT8FA!/b/dIUBAAAAAAAA&bo=KwRWAgAAAAARF1s!&rf=viewer_4)

最后再来看下更复杂 更高阶多项式的情况，这会让你得到非常复杂的决策边界。

![44.4](http://m.qpic.cn/psb?/V12umJF70r2BEK/2gucMaUnUE.YriIA3f3sCs*NOCrjplzruwvFuaQVKMI!/b/dN4AAAAAAAAA&bo=JgRMAgAAAAARB1w!&rf=viewer_4)



# Logistic Regression Model

## 课时45  代价函数(Cost Function)  10:23

集合logistic回归模型的参数θ

定义用来拟合参数的优化目标或者叫代价函数。
这就是logistic回归模型的拟合问题。

我们的每一个样本都用n+1维的特征向量表示。

对于给定的训练集我们如何选择 或者说如何拟合 参数θ
![45.1]()



## 课时46  简化代价函数与梯度下降(Simplified Cost Function and Gradient Descent)    10:14



## 课时47  高级优化(Advanced Optimization)  14:06




# Multiclass Classification

## 课时48  多元分类：一对多(Multiclass Classification:One vs all)  06:15



## 课时49  本章课程总结