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

为了更好地理解反向传播算法，我们再来仔细研究一下前向传播的原理：

前向传播算法：

![63.1](http://m.qpic.cn/psb?/V12umJF70r2BEK/md4NJfPaWTEppgbNIAyuVC4IU3BKgJHLxM6TbJTpyYQ!/b/dCIBAAAAAAAA&bo=KQO.AQAAAAARF7U!&rf=viewer_4)

在进行向前传播时，我们肯有些特定的样本,如$(x^{(i)},y^{(i)})$
我们把x^{(i)}放进输入层，他们时我们为输入层设置的值,当对其进行前向传播,传播到第一个隐藏层时,我们要计算出$z^{(2)}_1$和$z^{(2)}_2$，他们是输入单元的加权和，然后我们将sigmoid逻辑函数还有sigmoid激活函数应用到z值上,得到这些激活值$a^{(2)}_1$ 和 $a^{(2)}_2$,然后继续向前传播,最后得到$a^{(4)}_1$,也就是神经网络的最后的输出值



反向传播算法做法

![63.2](http://m.qpic.cn/psb?/V12umJF70r2BEK/uxFzNl5Oi5QB3r.CgNIpqODfFVpPo*RpXk65O5Tw.Y8!/b/dCEBAAAAAAAA&bo=LwPEAQAAAAARF8k!&rf=viewer_4)

我们关注一下下面的这个样本$x^{(i)}$和$y^{(i)}$，因为只有一个输出单元，所以忽略正则化项，剩下的代价函数对应了第i个训练样本,即代价函数所对应的训练样本$(x^{(i)},y^{(i)})$,所以第i对样本的代价函数可以写成$cost(i)$的形式,它扮演了一个类似方差的角色，你也可以把cost(i)看出下面的形式

$$cost(i) ≈ (h_\Theta(x^{(i)})-y^{(i)})^2 $$

即神经网络的输出值与实际值的方差

再回看反向传播的过程

![63.3](http://m.qpic.cn/psb?/V12umJF70r2BEK/mvEuMBCmZ4jn0UW.1DnoeRZKVe1j2*19KrXiFxGYtRQ!/b/dCIBAAAAAAAA&bo=RAPgAQAAAAARB5Y!&rf=viewer_4)

一种直观地理解是反向传播算法就是在计算$\delta^{(l)}_j$项，我们可以把它看作是我们在第l层中第j个单元中得到的激活项的“误差”.

更正式一点的说法是，$\delta^{(l)}_j$项实际上是代价函数$cost(i)$关于$z^{(l)}_j$的偏导数,也就是计算出的z项的加权和,或者说代价函数关于z项的偏导数,具体来说这个代价函数是一个关于标签y和神经网络中$h(x)$的输出值的函数.

$\delta$项实际上是代价函数关于这些所计算出的中间项的偏导数，他们衡量的是，为了影响这些中间值，我们想要改变神经网络中的权重的程度进而影响整个神经网络的输出$h(x)$并影响所有的代价函数

易知：
$\delta^{(4)}_1 = y^{(i)}-a^{(4)}_1$

求出输出项的误差值后对其进行反向传播,得出第3层的误差,然后传到第2层得到第2层的误差.

而反向传播的计算过程实际上就是后一层的$\delta$项的加权和由对应边的强度来进行加权,即

$$\delta^{(2)}_2 = \Theta^{(2)}_{12}\delta^{(3)}_1+\Theta^{(2)}_{22}\delta^{(3)}_2$$






## Backpropagation in Practice()

## 64、Implementation Note:Unrolling Parameters(使用注意：展开参数)

怎样把你的参数 从矩阵展开成向量 以便我们在高级最优化步骤中的使用需要 

![64.1](http://m.qpic.cn/psb?/V12umJF70r2BEK/o4bVG9TMXc4mz62Lz981oXDZ0OpVP81yCy1j98CzFBo!/b/dCIBAAAAAAAA&bo=WAPPAQAAAAARB6U!&rf=viewer_4)

具体来讲 你执行了代价函数costFunction 输入参数是theta 函数返回值是代价函数以及导数值 

然后你可以将返回值 传递给高级最优化算法fminunc 顺便提醒 fminunc并不是唯一的算法 你也可以使用别的优化算法 

但它们的功能 都是取出这些输入值 @costFunction 以及theta值的一些初始值 

并且这些程序 都假设theta 和这些theta初始值 都是**参数向量** 也许是n或者n+1阶 但它们都是向量 同时假设这个代价函数 第二个返回值 也就是gradient值 也是n阶或者n+1阶 所以它也是一个向量 这部分在我们使用逻辑回归的时候 运行顺利 但现在 对于神经网络 我们的参数将不再是 向量 而是矩阵了 

因此对于一个完整的神经网络 我们的参数矩阵为θ(1) θ(2) θ(3) 在Octave中我们可以设为 矩阵Theta1 Theta2 Theta3 类似的 这些梯度项gradient 也是需要得到的返回值 那么在之前的视频中 我们演示了如何计算 这些梯度矩阵 它们是D(1) D(2) D(3) 在Octave中 我们用矩阵D1 D2 D3来表示 

怎样取出这些矩阵 并且将它们展开成向量 以便它们最终 成为恰当的格式 能够传入这里的Theta 并且得到正确的梯度返回值gradient 

具体来说 假设我们有这样一个神经网络 其输入层有10个输入单元 隐藏层有10个单元 最后的输出层 只有一个输出单元 因此s1等于第一层的单元数 s2等于第二层的单元数 s3等于第三层的 单元个数 在这种情况下 矩阵θ的维度 和矩阵D的维度 将由这些表达式确定 比如说 θ(1)是一个10x11的矩阵 以此类推 

![64.2](http://m.qpic.cn/psb?/V12umJF70r2BEK/8xzoj1IzPMKWEwNuGGXywGsye.CaEE6vaMl4NtWdj00!/b/dCEBAAAAAAAA&bo=VwPMAQAAAAARF7k!&rf=viewer_4)

因此 在Octave中 如果你想将这些矩阵 转化为向量 那么你要做的 是取出你的Theta1 Theta2 Theta3 然后使用这段代码 这段代码将取出 三个θ矩阵中的所有元素 也就是说取出Theta1 的所有元素 Theta2的所有元素 Theta3的所有元素 然后把它们全部展开 成为一个很长的向量 

也就是thetaVec 

同样的 第二段代码 将取出D矩阵的所有元素 然后展开 成为一个长向量 被叫做DVec 最后 如果你想从向量表达 返回到矩阵表达式的话 

你要做的是 比如想再得到Theta1 那么取thetaVec 抽出前110个元素 因此 Theta1就有110个元素 因为它应该是一个10x11的矩阵 所以 抽出前110个元素 然后你就可以 reshape矩阵变维命令来重新得到Theta1 同样类似的 要重新得到Theta2矩阵 你需要抽出下一组110个元素并且重新组合 然后对于Theta3 你需要抽出最后11个元素 然后执行reshape命令 重新得到Theta3 


为了使这个过程更形象 下面我们来看怎样将这一方法 应用于我们的学习算法 

假设说你有一些 初始参数值 θ(1) θ(2) θ(3) 我们要做的是 取出这些参数并且将它们 展开为一个长向量 我们称之为initialTheta 然后作为theta参数的初始设置 传入函数fminunc 

我们要做的另一件事是执行代价函数costFunction 

实现算法如下 

![64.3](http://m.qpic.cn/psb?/V12umJF70r2BEK/W3.0B0TmGhQ6Gell5BF74jQ.HLHp4vjHsZebAxyyV7s!/b/dCIBAAAAAAAA&bo=ZQNtAQAAAAARFyo!&rf=viewer_4)

代价函数costFunction 将传入参数thetaVec 这也是包含 我所有参数的向量 是将所有的参数展开成一个向量的形式 

因此我要做的第一件事是 我要使用 thetaVec和重组函数reshape 因此我要抽出thetaVec中的元素 然后重组 以得到我的初始参数矩阵 θ(1) θ(2) θ(3) 所以这些是我需要得到的矩阵 因此 这样我就有了 一个使用这些矩阵的 更方便的形式 这样我就能执行前向传播 和反向传播 来计算出导数 以求得代价函数的J(θ) 

最后 我可以取出这些导数值 然后展开它们 让它们保持和我展开的θ值 同样的顺序 我要展开D1 D2 D3 来得到gradientVec 这个值可由我的代价函数返回 它可以以一个向量的形式返回这些导数值 

现在 我想 对怎样进行参数的矩阵表达式 和向量表达式 之间的转换 有了一个更清晰的认识 

使用矩阵表达式 的好处是 当你的参数以矩阵的形式储存时 你在进行正向传播 和反向传播时 你会觉得更加方便 当你将参数储存为矩阵时 一大好处是 充分利用了向量化的实现过程 

相反地 向量表达式的优点是 如果你有像thetaVec或者DVec这样的矩阵 当你使用一些高级的优化算法时 这些算法通常要求 你所有的参数 都要展开成一个长向量的形式 希望通过我们刚才介绍的内容 你能够根据需要 更加轻松地 在两种形式之间转换



## 65、Gradient Checking(梯度检测)

当我们对一个较为复杂的模型（例如神经网络）使用梯度下降算法时，可能会存在一些不容易察觉的错误，意味着，虽然代价看上去在不断减小，但最终的结果可能并不是最优解。

为了避免这样的问题，我们采取一种叫做梯度的数值检验（Numerical Gradient Checking）方法。这种方法的思想是通过估计梯度值来检验我们计算的导数值是否真的是我们要求的。

对梯度的估计采用的方法是在代价函数上沿着切线的方向选择离两个非常近的点然后计算两个点的平均值用以估计梯度。即对于某个特定的 $\theta$，我们计算出在 $\theta$-$\varepsilon $ 处和 $\theta$+$\varepsilon $ 的代价值（$\varepsilon $是一个非常小的值，通常选取 0.001），然后求两个代价的平均，用以估计在 $\theta$ 处的代价值。

当$\theta$是一个向量时，我们则需要对偏导数进行检验。因为代价函数的偏导数检验只针对一个参数的改变进行检验，下面是一个只针对$\theta_1$进行检验的示例： $$ \frac{\partial}{\partial\theta_1}=\frac{J\left(\theta_1+\varepsilon_1,\theta_2,\theta_3...\theta_n \right)-J \left( \theta_1-\varepsilon_1,\theta_2,\theta_3...\theta_n \right)}{2\varepsilon} $$

最后我们还需要对通过反向传播方法计算出的偏导数进行检验。

根据上面的算法，计算出的偏导数存储在矩阵 $D_{ij}^{(l)}$ 中。检验时，我们要将该矩阵展开成为向量，同时我们也将 $\theta$ 矩阵展开为向量，我们针对每一个 $\theta$ 都计算一个近似的梯度值，将这些值存储于一个近似梯度矩阵中，最终将得出的这个矩阵同 $D_{ij}^{(l)}$ 进行比较。


## 66、Random Initialization(随机初始化)

任何优化算法都需要一些初始的参数。到目前为止我们都是初始所有参数为0，这样的初始方法对于逻辑回归来说是可行的，但是对于神经网络来说是不可行的。如果我们令所有的初始参数都为0，这将意味着我们第二层的所有激活单元都会有相同的值。同理，如果我们初始所有的参数都为一个非0的数，结果也是一样的。

我们通常初始参数为正负ε之间的随机值，假设我们要随机初始一个尺寸为10×11的参数矩阵，代码如下：

Theta1 = rand(10, 11) * (2*eps) – eps





## 67、Putting It Together(组合到一起))

小结一下使用神经网络时的步骤：

网络结构：第一件要做的事是选择网络结构，即决定选择多少层以及决定每层分别有多少个单元。

第一层的单元数即我们训练集的特征数量。

最后一层的单元数是我们训练集的结果的类的数量。

如果隐藏层数大于1，确保每个隐藏层的单元个数相同，通常情况下隐藏层单元的个数越多越好。

我们真正要决定的是隐藏层的层数和每个中间层的单元数。

训练神经网络：

参数的随机初始化

利用正向传播方法计算所有的$h_{\theta}(x)$

编写计算代价函数 $J$ 的代码

利用反向传播方法计算所有偏导数

利用数值检验方法检验这些偏导数

使用优化算法来最小化代价函数




## Application of Neural Networks(神经网络的应用)

## 68、Autonomous Driving(无人驾驶)






## Review