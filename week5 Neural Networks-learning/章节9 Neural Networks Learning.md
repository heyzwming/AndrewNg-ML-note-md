# 九、Neural Networks:Learning
===

## Cost Function and Backpropagation(代价函数与反向传播))

## 61、Cost Function(代价函数)

假设我们有一个与左图类似的神经网络结构，再假设我们有一个像这样的训练集，其中有m组训练样本$(x^{(i)},y^{(i)})$

$L$ = 神经网络结构的总层数$(L = 4)$  
$S_l$ = 第$L$层的单元数，也就是神经元的数量(不包括第L层的偏差单元)($S_1 = 3，S_2 = 5$,$S_4 = S_l = 4$)  

![61.1](http://m.qpic.cn/psb?/V12umJF70r2BEK/QQbEVzJb8xAlZquj7PmZkAKzukBbcuYFfINIWFEb254!/b/dCEBAAAAAAAA&bo=RgPEAQAAAAARB7A!&rf=viewer_4)

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

$$J(\theta) = -\frac{1}{m} \left[
    \sum_{i=1}^m \sum_{k=1}^K y^{(i)}_k log(h_\theta(x^{(i)}))_k+(1-y^{(i)}_k)log(1-(h_\Theta(x^{(i)}))_k)
\right] + \frac{\lambda}{2m}\sum_{l=1}^{L-1} \sum_{i=1}^{S_l} \sum_{j=1}^{S_l+1} (\Theta^{(l)}_{ji})^2
$$

最末尾附加的项就是类似于我们在逻辑回归里所用的正则化项



## 62、Backpropagation Algorithm(反向传播算法)

之前我们在计算神经网络预测结果的时候我们采用了一种正向传播方法。

![62.1正向传播](http://m.qpic.cn/psb?/V12umJF70r2BEK/cLUx0dqhZdUpiPufzrf5fluyvYnalAw*m004bcwFtJw!/b/dCEBAAAAAAAA&bo=7QKZAQAAAAARB0c!&rf=viewer_4)

我们从第一层开始正向一层一层进行计算$J(\theta)$和偏导项$\frac{\partial}{\partial\Theta^{(l)}_{ij}}J\left(\Theta\right)$，直到最后一层的$h_{\theta}\left(x\right)$。

现在，为了计算代价函数的偏导数$\frac{\partial}{\partial\Theta^{(l)}_{ij}}J\left(\Theta\right)$，我们需要采用一种反向传播算法，也就是首先计算最后一层的误差，然后再一层一层反向求出各层的误差，直到倒数第二层。 以一个例子来说明反向传播算法。

从直观上说就是对每一个结点，我们计算这样一项$\delta{(l)}_j$,代表了第l层第j个结点的激活值误差

![62.2](http://m.qpic.cn/psb?/V12umJF70r2BEK/ODJthqdaJ*IqSkYpQ73bkRYLD839rLHmjTFzoBbSW5I!/b/dCIBAAAAAAAA&bo=HwPAAQAAAAARF*0!&rf=viewer_4)

我们用右边这个有四层的神经网络结构做例子,所以这里的L是4,对每一个输入单元我们都要计算$\delta$项,所以最后一层(第四层)的第j个单元的$\delta^{(4)}_j = a^{(4)}_j - y_j$,即这个单元的激活值(假设$h_\Theta(x)$的输出值)减去训练样本里的真实值.

接下来我们要计算网络中前面几层的误差项$\delta$

下面这个就是计算$\delta$的公式

$\delta^{(4)}=a^{(4)}-y$ 

我们利用这个误差值来计算前一层的误差：

$$\delta^{(3)}=\left({\Theta^{(3)}}\right)^{T}\delta^{(4)}·\ast g'\left(z^{(3)}\right)$$

$$\delta^{(2)}=\left({\Theta^{(2)}}\right)^{T}\delta^{(3)}·\ast g'\left(z^{(2)}\right)$$

其中 $g'(z^{(3)})$是 $S$ 形函数的导数，$g'(z^{(3)})=a^{(3)}\ast(1-a^{(3)})$。而$(θ^{(3)})^{T}\delta^{(4)}$则是权重导致的误差的和。

下一步是继续计算第二层的误差： $\delta^{(2)}=(\Theta^{(2)})^{T}\delta^{(3)}\ast g'(z^{(2)})$ 

因为第一层是输入变量，不存在误差。我们有了所有的误差的表达式后，便可以计算代价函数的偏导数了，假设$λ=0$，即我们不做任何正则化处理时有： $\frac{\partial}{\partial\Theta_{ij}^{(l)}}J(\Theta)=a_{j}^{(l)} \delta_{i}^{l+1}$

反向传播法这个名字源于我们从输出层开始计算$\delta$项,然后我们返回到上一层计算第三隐藏层$\delta$项,接着我们再往前一步来计算$\delta(2)$,我们是类似于把输出层的误差反向传播给了第3层,然后再传到第二层，这就是反向传播的意思.

重要的是清楚地知道上面式子中上下标的含义：

$l$ 代表目前所计算的是第几层。

$j$ 代表目前计算层中的激活单元的下标，也将是下一层的第$j$个输入变量的下标。

$i$ 代表下一层中误差单元的下标，是受到权重矩阵中第$i$行影响的下一层中的误差单元的下标。


假设我们有一个m个样本的训练集，我们先要固定这些带下标ij的$\Delta_{ij}^{(l)}$,他们会被用来计算$\frac{\partial}{\partial\Theta_{ij}^{(l)}}J(\Theta)$偏导项,这些$\delta$会被作为累加项慢慢地增加,以算出这些偏导数.

接下来我们将循环遍历我们的训练集

![62.3](http://m.qpic.cn/psb?/V12umJF70r2BEK/GzKe0kC1XT.KAPOZwgVzh8FBYcYNa5yBcxCIInA8KXk!/b/dCEBAAAAAAAA&bo=BwOrAQAAAAARF44!&rf=viewer_4)

我们将取训练样本$(x^{(i)},y^{(i)})$,然后设置$a^{(1)}$也就是输入层的激活函数为$x^{(i)}$(第i个训练样本的输入值),我们先用正向传播算法运算出所有的激活项$a^{(l)}$。然后用我们这个样本的输出值$y^{(i)}$来计算这个输出值所对应的误差项$\delta(L) = a^{(L)}-y^{(i)}$,最后运用反向传播算法计算预测结果与训练集结果的误差$\delta^{(L-1)},\delta^{(L-2)},...,\delta^{(2)}$,没有$\delta^{(2)}$因为我们不需要对输入层考虑误差项，然后利用该误差运用反向传播法($\Delta^{(l)}_{ij} := \Delta^{(l)}_{ij}+a^{(l)}_j\delta^{(l+1)}_i$)计算出直至第二层的所有误差。当然也可以把最后一步的算法改成向量形式$\Delta^{(l)} := \Delta^{(l)}+\delta^{(l+1)}(a^{(l)})^T$

在求出了$\Delta_{ij}^{(l)}$之后，我们便可以计算代价函数的偏导数了，计算方法如下：

$D_{ij}^{(l)} :=\frac{1}{m}\Delta_{ij}^{(l)}+\lambda\Theta_{ij}^{(l)}$ ${if}; j \neq 0$

$D_{ij}^{(l)} :=\frac{1}{m}\Delta_{ij}^{(l)}$ ${if}; j = 0$

最后的最后

$$\frac{\partial}{\partial\Theta_{ij}^{(l)}}J(\Theta) = D^{(l)}_{ij}$$

在Octave 中，如果我们要使用 fminuc这样的优化算法来求解求出权重矩阵，我们需要将矩阵首先展开成为向量，在利用算法求出最优解后再重新转换回矩阵。 




## 63、Backpropagation Intuition(理解反向传播)






## Backpropagation in Practice()

## 64、Implementation Note:Unrolling Parameters(使用注意：展开参数)






## 65、Gradient Checking(梯度检测)






## 66、Random Initialization(随机初始化)







## 67、Putting It Together(组合到一起))






## Application of Neural Networks(神经网络的应用)

## 68、Autonomous Driving(无人驾驶)






## Review