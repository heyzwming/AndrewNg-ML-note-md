# 九、Neural Networks:Learning
===

## Cost Function and Backpropagation(代价函数与反向传播))

## 61、Cost Function(代价函数)

假设我们有一个与左图类似的神经网络结构，再假设我们有一个像这样的训练集，其中有m组训练样本$(x^{(i)},y^{(i)})$

$L$ = 神经网络结构的总层数$(L = 4)$  
$S_l$ = 第$L$层的单元数，也就是神经元的数量(不包括第L层的偏差单元)($S_1 = 3，S_2 = 5$,$S_4 = S_l = 4$)  

![61.1]()

我们会考虑两种分类问题  
* 第一种是二元分类，这里的y只能是0或者1，在这种情况下我们会有一个输出单元，  
在这种情况下$S_l$是输出单元的个数，其中的$L$同样代表最后一层的序号，因为这就是这个网络结构种的层数，所以我们在输出层中单元数目就将是1，为了方便我们把K设置为1，你也可以把K当成输出层的单元数目

* 第二种是多类别分类问题，也就是说会有K个不同的类和输出单元，我们的假设会输出K维向量同时输出单元的个数$S_l=K$ 

接下来我们要定义代价函数：

$$J(\theta) = -\frac{1}{m} \begin{bmatrix}
    \sum_{i=1}^m y^{(i)}logh_\theta(x^{(i)}+(1-y^{(i)})log(1-h_\theta(x^{(i)}))
\end{bmatrix} + \frac{\lambda}{2m}\sum_{j=1}^n\theta^2_j
$$

我们在神经网络中使用的代价函数其实是逻辑回归中使用的代价函数的一般形式，对逻辑回归函数来说我们通常使代价函数$J(\theta)$最小化

与原本的逻辑回归不同的是我们对每一个代价函数都有K个输入单元。

我们用$(h_\Theta(x))_i$来表示第i个输出，$h(x)$是一个K维向量，下标i表示选择输出神经网络输出向量中的第i个元素。

现在的代价函数

$$J(\theta) = -\frac{1}{m} \begin{bmatrix}
    \sum_{i=1}^m \sum_{k=1}^K y^{(i)}_k log(h_\theta(x^{(i)}))_k+(1-y^{(i)}_k)log(1-(h_\Theta(x^{(i)}))_k)
\end{bmatrix} + \frac{\lambda}{2m}\sum_{l=1}^{L-1} \sum_{i=1}^{S_l} \sum_{j=1}^{S_l+1} (\Theta^{(l)}_{ji})^2
$$

最末尾附加的项就是类似于我们在逻辑回归里所用的正则化项



## 62、Backpropagation Algorithm(反向传播算法)






## 63、Backpropagation Intuition(理解反向传播)






## Backpropagation in Practice()

## 64、Implementation Note:Unrolling Parameters(使用数以：展开参数)






## 65、Gradient Checking(梯度检测)






## 66、Random Initialization(随机初始化)







## 67、Putting It Together(组合到一起))






## Application of Neural Networks(神经网络的应用)

## 68、Autonomous Driving(无人驾驶)






## Review