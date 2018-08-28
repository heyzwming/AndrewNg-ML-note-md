八、Neural Networks:Representation
===

## Motivations

## 54、Non-linear Hypotheses ( 非线性假设 )

看几个需要学习复杂的非线性假设的例子

假设有个监督学习分类问题，如果使用逻辑回归来解决这个问题，你可以构造出一个包含很多非线性项的逻辑回归函数，这里的g仍然是sigmoid函数，如果只有x1 x2两个特征，你确实可以用多项式的逻辑回归得到不错的结果，但事实上一般性的问题会有很多的特征。

现在假设你有一个包含3个特征量的训练集，现在你想要建立一个关于这个训练集的二次假设方程：
$$g(\theta_0+\theta_1x_1^2+\theta_2x_1x_2+\theta_3x_1x_3+\theta_4x_2^2+\theta_5x_2x_3+\theta_6x_3^2)$$

在这样的情况下，你需要两两配对各种情况，需要$​\frac{(3+2–1)!}{(2!⋅(3−1)!)}​$个二次项。

进一步，如果你有一个包含100个特征量的训练集，那么你需要​$\frac{(100+2–1)!}{(2⋅(100−1)!)}$=$5050​$个二次项。

我们继续推广，可以发现，二次项的空间复杂度为$​O(n^2/2)$​ ；如果再进一步推广到三次项，那么空间复杂度为$​O(n^3)​$ 。可以看到，当初始特征个数n很大时将这些高阶多项式项数包括到特征里会使特征空间急剧膨胀，当特征个数n很大时，增加特征来建立非线性分类器并不是一个好做法。这是一个非常陡峭的增长函数，如果我们需要处理大量特征量、高次假设方程，那么空间的增长将是巨大的。

因此，我们需要寻找一个替代的方法，优化如此巨大的复杂度。

![54.1](http://m.qpic.cn/psb?/V12umJF70r2BEK/51e73VLVCcwQR5B1.c5.gyTPr38Qr7NoT5n4LosVEE0!/b/dIUBAAAAAAAA&bo=pQULAwAAAAARF4g!&rf=viewer_4)

对于许多实际的机器学习问题，特征个数n是很大的。


对于一个图像识别问题中输入的图像是否是我们要识别的物体，我们需要一个非线性假设来区分开两类样本

![54.2](http://a2.qpic.cn/psb?/V12umJF70r2BEK/o2MvlvfOkI2x6Y1*LMbbkm9kDJpElERRUKXWg4q5V8I!/b/dA0BAAAAAAAA&ek=1&kp=1&pt=0&bo=lwUSAwAAAAARF6M!&tl=3&vuin=904260897&tm=1535421600&sce=60-2-2&rf=viewer_4)

只是包括平方项或者立方项特征简单的logistic回归算法并不是一个在n很大时学习复杂的非线性假设的好办法，因为特征过多

## 55、Neurons and the Brain ( 神经元与大脑 )

我们知道，目前大脑拥有最厉害的“机器学习”算法，那我们能否模仿它来实现更好的机器学习算法呢？神经网络可以在一定程度上模仿大脑的运作机制。这一领域早就已经有了很多概念，不过因为运算量大而难以进一步发展；近年来，随着计算机速率的提升，神经网络也得到了发展的动力，再次成为热点。

我们对于不同的数据处理需求可能会提出各种不同的算法，那么，有没有可能所有需求都由一个算法来实现，就像物理一样，人类追求一个大一统的“统一场论”呢？

科学家尝试做过这么一个实验：将大脑中听觉皮层与耳朵之间的神经连接切断，并且将听觉皮层与眼睛相连，结果发现听觉皮层可以“看到”东西。

![55.1](http://m.qpic.cn/psb?/V12umJF70r2BEK/wTVxMxPsgVRcA5gVALj**QNyJB*c2pqrMApJroKOgtA!/b/dN4AAAAAAAAA&bo=HAOlAQAAAAARB4s!&rf=viewer_4)

这说明，统一的学习算法是可能实现的。

## Neural Networks



## 56、Model Representation Ⅰ ( 模型展示 Ⅰ )

我们在应用神经网络的时候如何表示我们的假设或模型？

神经网络模仿了大脑中的神经元或者神经网络.

下面是一个简单的例子，图中的神经元是一个基本的运算单元，它由电信号从多个接受通道获取一定数目的信息输入(树突dendrites),并且计算后通过唯一的输出通道(轴突axon)给出输出。

![56.1](http://m.qpic.cn/psb?/V12umJF70r2BEK/g3RCoPG1dw7Wuzv379hTmENvJoknOGcqWV4*SA*9Vwk!/b/dN8AAAAAAAAA&bo=bgSgAgAAAAARB*g!&rf=viewer_4)

我们的神经网络模型，便是模仿了这一过程。

黄色的圆圈是一个类似神经元细胞体的东西,然后我们通过树突或者说输入通道，传递给它一些信息,然后神经元做一些计算,通过输出通道输出假设函数$h_\theta(x)$计算结果

我们的**输入**是​$x_1⋯x_n$​ ，**输出**是假设函数。额外的，我们需要添加​$x_0=1$​，称作**偏置单元(bias unit)或偏置神经元**。

在神经网络的分类算法中，我们使用相同的逻辑函数$\frac{1}{1+e^{−θ^Tx}}​​$，有时候，我们称它为**逻辑激活函数**（Sigmoid/logistic activation function）。其中的**激活函数**（activation function）一般来说指“由……驱动的神经元计算函数g(z)”，像上例就是“由逻辑函数驱动的神经元计算函数g(z)”。

这里的参数θ也被称作模型的参数或权重（weight）。


![56.2](http://m.qpic.cn/psb?/V12umJF70r2BEK/IOB2yFyXuHRWLEv1ILrvsbMVYT6blwWiA8V7*lTt1zc!/b/dN4AAAAAAAAA&bo=IwXZAgAAAAARF90!&rf=viewer_4)
在上面的模型中，红色的圆圈代表了一个神经元。在实际上，神经网络就是由不同的神经元连接组合而来的：

![56.3](http://m.qpic.cn/psb?/V12umJF70r2BEK/FZnR3uBS5abUpTLtUv6i2qJeMpWPSfsVTfu5kN.Nfu0!/b/dGwBAAAAAAAA&bo=IQXgAgAAAAARF.Y!&rf=viewer_4)

可以看到，我们的输入层（第一层）接收了数据的输入，计算后输出到输出层（第三层），但是这个模型中，中间还多了一层数据的处理，这一层是由输入层的数据加权组合后重新映射成的，称为**隐藏层（Hidden Layer）**。

一个简单的表达形式如下：

⎡⎣⎢x0x1x2⎤⎦⎥→[   ]→hθ(x)
$$\begin{bmatrix}
    x_0 \\ x_1 \\ x_2
\end{bmatrix} -> \begin{bmatrix}
    & & &
\end{bmatrix}  -> h_\theta(x)$$

一般而言，我们把隐藏层的节点或者说中间节点，称作**“激活单元”（activation units）**，并且有如下符号：

$a^{(j)}_i$=第j层的第i个激活单元  
$\Theta^{(j)}$=控制从第j层到j+1层的映射函数的权重矩阵

如果我们有一个隐藏层，那么整个流程看起来是这样的：

$$\begin{bmatrix}
    x_0 \\ x_1 \\ x_2 \\ x_3 
\end{bmatrix} ->
\begin{bmatrix}
    x_1^{(2)} \\ x_2^{(2)} \\ x_3^{(2)} 
\end{bmatrix} -> h_\theta(x)
$$


其中每一个激活节点的值是这样计算的：

a(2)1=g(Θ(1)10x0+Θ(1)11x1+Θ(1)12x2+Θ(1)13x3)a(2)2=g(Θ(1)20x0+Θ(1)21x1+Θ(1)22x2+Θ(1)23x3)a(2)3=g(Θ(1)30x0+Θ(1)31x1+Θ(1)32x2+Θ(1)33x3)hΘ(x)=a(3)1=g(Θ(2)10a(2)0+Θ(2)11a(2)1+Θ(2)12a(2)2+Θ(2)13a(2)3)
也就是说，第j层的权重矩阵的每一行对应一个加权组合，然后通过g(z)函数映射到j+1层的节点。

由矩阵乘法可知：因此如果要从含有​$s_j$​个单元的第j层映射到含有$s_{j+1}$个单元的第j+1层，那么权重矩阵​Θ(j)​的尺寸为​$s_{j+1}×(s_j+1)$​，其中的+1是因为要考虑偏置单元​$x_0$​。


## 57、Model Representation Ⅱ ( 模型展示 Ⅱ )



## Applications



## 58、Examples and Intuitions Ⅰ ( 例子与直觉理解 Ⅰ )



## 59、Examples and Intuitions Ⅱ ( 例子与直觉理解 Ⅱ )



## 60、Multiclass Classification ( 多元分类 )


