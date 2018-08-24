Logistic 回归(Logistic Regression)
===


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




# Logistic Regression Model

## 课时45  代价函数(Cost Function)  10:23



## 课时46  简化代价函数与梯度下降(Simplified Cost Function and Gradient Descent)    10:14



## 课时47  高级优化(Advanced Optimization)  14:06




# Multiclass Classification

## 课时48  多元分类：一对多(Multiclass Classification:One vs all)  06:15



## 课时49  本章课程总结