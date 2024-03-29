十七、Large Scale Machine learning(大规模机器学习)
===
## Gradient Dsecent with Large Datasets(大数据集的梯度下降)
---
## 17.1、 Learning With Large Datasets(学习大数据集)

Andrew Ng在这里引用了一句话：“It’s not who has the best algorithm that wins. It’s who has the most data.”从这句话引申出了今天的主题–海量数据下的梯度下降算法。

![learning with large datasets]()

在之前的学习中我们知道，如果在算法的训练样本量m不足的时候得到的模型具有高方差（High Variance），那么此时我们需要更多的训练样本。但是如果算法具有高偏差，提高样本数量并不会显著改善模型的性能。

在今天，数据集很容易就可以达到m=100,000,000这样的数量级（比如人口调查、通讯消息等）。在这样的背景下，梯度下降算法每一步都需要汇总上亿个训练样本来完成一次迭代。那么我们能不能通过一些方法来分流如此大量的计算呢？

## 17.2、 Stochastic Gradient Descent(随机梯度下降)

随机梯度下降算法相对于传统（或者说批处理）梯度下降算法而言，在海量数据的应用场景下有更高的效率和扩展性。

随机梯度下降算法的代价函数可以用一个有区别但是相通的方式来表达：

$$cost(θ,(x(i),y(i)))=\frac{1}{2}​(h_θ​(x^{(i)})−y^{(i)})^2$$

唯一的区别在于上面的代价函数消除了m常量。

$$J_{train}​(θ)=\frac{1}{m}\sum_{i=1}^m​cost(θ,(x^{(i)},y^{(i)}))$$
​$J_{train}​$现在表达的意义是训练样本集的平均代价。



相对于传统梯度下降算法而言，随机梯度下降算法的步骤如下：

随机的洗牌（shuffle）数据集。（为了加快收敛速度）
对于i=1,…,m

$$Θ_j​ := Θ_j​−α(h_Θ​(x^{(i)})−y^{(i)})⋅x^{(i)}_j​$$

重复第二步1-10次
该算法在一个时刻仅仅会拟合一个训练样本。这样一来，梯度下降过程中就不需要先扫描全部m个训练样本了。随机梯度下降算法不太可能收敛在全局最小值处，而是会随机徘徊，但通常会产生足够接近的结果。随机梯度下降算法通常需要1-10次遍历数据集来接近全局最小值。

![Batch gradient descent]()

![Stochastic gradient descent]()

（这个算法的作用就是，不再是一次性读完所有的东西再一次性完成一轮参数学习，而是一点一点地动态学习。也正是因为如此，随机梯度下降算法的收敛是在逐步向最优解徘徊）



---
对于很多机器学习算法 包括线性回归、逻辑回归、神经网络等等 算法的实现都是通过得出某个代价函数 或者某个最优化的目标来实现的 然后使用梯度下降这样的方法来求得代价函数的最小值 当我们的训练集较大时 梯度下降算法则显得计算量非常大 在这段视频中 我想介绍一种跟普通梯度下降不同的方法 随机梯度下降(stochastic gradient descent) 用这种方法我们可以将算法运用到较大训练集的情况中 假如你要使用梯度下降法来训练某个线性回归模型 简单复习一下 我们的假设函数是这样的 代价函数是你的假设在训练集样本上预测的平均平方误差的二分之一倍的和 通常我们看到的代价函数都是像这样的弓形函数 因此 画出以θ0和θ1为参数的代价函数J 就是这样的弓形函数 这就是梯度下降算法 在内层循环中 你需要用这个式子反复更新参数θ的值 在这段视频剩下的时间里 我将依然以线性回归为例 但随机梯度下降的思想也可以应用于其他的学习算法 比如逻辑回归、神经网络或者其他依靠梯度下降来进行训练的算法中 这张图表示的是梯度下降的做法 假设这个点表示了参数的初始位置 那么在你运行梯度下降的过程中 多步迭代最终会将参数锁定到全局最小值 迭代的轨迹看起来非常快地收敛到全局最小 而梯度下降法的问题是 当m值很大时 计算这个微分项的计算量就变得很大 因为需要对所有m个训练样本求和 所以假如m的值为3亿 美国就有3亿人口 美国的人口普查数据就有这种量级的数据记录 所以如果想要为这么多数据拟合一个线性回归模型的话 那就需要对所有这3亿数据进行求和 这样的计算量太大了 这种梯度下降算法也被称为批量梯度下降(batch gradient descent) “批量”就表示我们需要每次都考虑所有的训练样本 我们可以称为所有这批训练样本 也许这不是个恰当的名字 但做机器学习的人就是这么称呼它的 想象一下 如果你真的有这3亿人口的数据存在硬盘里 那么这种算法就需要把所有这3亿人口数据读入计算机 仅仅就为了算一个微分项而已 你需要将这些数据连续传入计算机 因为计算机存不下那么大的数据量 所以你需要很慢地读取数据 然后计算一个求和 再来算出微分 所有这些做完以后 你才完成了一次梯度下降的迭代 然后你又需要重新来一遍 也就是再读取这3亿人口数据 做个求和 然后做完这些 你又完成了梯度下降的一小步 然后再做一次 你得到第三次迭代 等等 所以 要让算法收敛 绝对需要花很长的时间 相比于批量梯度下降 我们介绍的方法就完全不同了 这种方法在每一步迭代中 不用考虑全部的训练样本 只需要考虑一个训练样本 在开始介绍新的算法之前 我把批量梯度下降算法再写在这里 这里是代价函数 这里是迭代的更新过程 梯度下降法中的这一项 是最优化目标 代价函数Jtrain(θ) 关于参数θj的偏微分 下面我们来看对大量数据来说更高效的这种方法 为了更好地描述随机梯度下降算法 代价函数的定义有一点区别 我们定义参数θ 关于训练样本(x(i),y(i))的代价 等于二分之一倍的 我的假设h(x(i))跟实际输出y(i)的误差的平方 因此这个代价函数值实际上测量的是我的假设在某个样本(x(i),y(i))上的表现 你可能已经发现 总体的代价函数Jtrain可以被写成这样等效的形式 Jtrain(θ)就是我的假设函数 在所有m个训练样本中的每一个样本(x(i),y(i))上的代价函数的平均值 用这样的方法应用到线性回归中 我来写出随机梯度下降的算法 随机梯度下降法的第一步是将所有数据打乱 我说的随机打乱的意思是 将所有m个训练样本重新排列 这就是标准的数据预处理过程 稍后我们再回来讲 随机梯度下降的主要算法如下 在i等于1到m中进行循环 也就是对所有m个训练样本进行遍历 然后进行如下更新 我们按照这样的公式进行更新 θj等于θj减α乘以h(x(i))减y(i)乘以x(i)j 同样还是对所有j的值进行更新 不难发现 这一项实际上就是我们批量梯度下降算法中 求和式里面的那一部分 事实上 如果你数学比较好的话 你可以证明这一项 也就是这一项 是等于这个cost函数关于参数θj的偏微分 这个cost函数就是我们之前先定义的代价函数 最后画上大括号结束算法的循环 随机梯度下降的做法实际上就是扫描所有的训练样本 首先是我的第一组训练样本(x(1),y(1)) 然后只对这第一个训练样本 对它的代价函数 计算一小步的梯度下降 换句话说 我们要关注第一个样本 然后把参数θ稍微修改一点 使其对第一个训练样本的拟合变得好一点 完成这个内层循环以后 再转向第二个训练样本 然后还是一样 在参数空间中进步一小步 也就是稍微把参数修改一点 然后让它对第二个样本的拟合更好一点 做完第二个 再转向第三个训练样本 同样还是修改参数 让它更好的拟合第三个训练样本 以此类推 直到完成所有的训练集 然后外部这个重复循环会多次遍历整个训练集 从这个角度分析随机梯度下降算法 我们能更好地理解为什么一开始要随机打乱数据 这保证了我们在扫描训练集时 我们对训练集样本的访问是随机的顺序 不管你的数据是否已经随机排列过 或者一开始就是某个奇怪的顺序 实际上这一步能让你的随机梯度下降稍微快一些收敛 所以为了保险起见 最好还是先把所有数据随机打乱一下 如果你不知道是否已经随机排列过的话 但随机梯度下降的更重要的一点是 跟批量梯度下降不同 随机梯度下降不需要等到对所有m个训练样本 求和来得到梯度项 而是只需要对单个训练样本求出这个梯度项 我们已经在这个过程中开始优化参数了 就不用等到把所有那3亿的美国人口普查的数据拿来遍历一遍 不需要等到对所有这些数据进行扫描 然后才一点点地修改参数 直到达到全局最小值 对随机梯度下降来说 我们只需要一次关注一个训练样本 而我们已经开始一点点把参数朝着全局最小值的方向进行修改了 这里把这个算法再重新写一遍 第一步是打乱数据 第二步是算法的关键 是关于某个单一的训练样本(x(i),y(i))来对参数进行更新 让我们来看看 这个算法是如何更新参数θ的 之前我们已经看到 当使用批量梯度下降的时候 需要同时考虑所有的训练样本数据 批量梯度下降的收敛过程 会倾向于一条近似的直线 一直找到全局最小值 与此不同的是 在随机梯度下降中 每一次迭代都会更快 因为我们不需要对所有训练样本进行求和 每一次迭代只需要保证对一个训练样本拟合好就行了 所以 如果我们从这个点开始进行随机梯度下降的话 第一次迭代 可能会让参数朝着这个方向移动 然后第二次迭代 只考虑第二个训练样本 假如很不幸 我们朝向了一个错误的方向 第三次迭代 我们又尽力让参数修改到拟合第三组训练样本 可能最终会得到这个方向 然后再考虑第四个训练样本 做同样的事 然后第五第六第七 等等 在你运行随机梯度下降的过程中 你会发现 一般来讲 参数是朝着全局最小值的方向被更新的 但也不一定 所以看起来它是以某个比较随机、迂回的路径在朝全局最小值逼近 实际上 你运行随机梯度下降 和批量梯度下降 两种方法的收敛形式是不同的 实际上随机梯度下降是在某个靠近全局最小值的区域内徘徊 而不是直接逼近全局最小值并停留在那点 但实际上这并没有多大问题 只要参数最终移动到某个非常靠近全局最小值的区域内 只要参数逼近到足够靠近全局最小值 这也会得出一个较为不错的假设 所以 通常我们用随机梯度下降法 也能得到一个很接近全局最小值的参数 对于绝大部分实际应用的目的来说 已经足够了 最后一点细节 在随机梯度下降中 我们有一个外层循环 它决定了内层循环的执行次数 所以 外层循环应该执行多少次呢 这取决于训练样本的大小 通常一次就够了 最多到10次 是比较典型的 所以我们可以循环执行内层1到10次 因此 如果我们有非常大量的数据 比如美国普查的人口数据 我说的3亿人口数据的例子 所以每次你只需要考虑一个训练样本 这里的i就是从1到3亿了 所以可能你每次只需要考虑一个训练样本 你就能训练出非常好的假设 这时 由于m非常大 那么内循环只用做一次就够了 但通常来说 循环1到10次都是非常合理的 但这还是取决于你训练样本的大小 如果你跟批量梯度下降比较一下的话 批量梯度下降在一步梯度下降的过程中 就需要考虑全部的训练样本 所以批量梯度下降就是这样微小的一次次移动 这也是为什么随机梯度下降法要快得多 这就是随机梯度下降了 如果你应用它 应该就能在很多学习算法中应用大量数据了 并且会得到更好的算法表现


## 17.3、 Mini-Batch Gradient Descent(Mini-Batch梯度下降)

这里直接给出三种梯度下降算法的不同：

批处理梯度下降算法；在每一次迭代中都使用了所有m个训练样本
随机梯度下降算法：在每一次迭代中只使用1个训练样本
最小批处理梯度下降算法：在每一次迭代中使用了b个训练样本
例如，对于总样本量为m,批大小为b的算法，重复如下操作：

for i = 1,11,21,31,…,991

$$θ_j​:=θ_j​−α\frac{1}{10}\sum_{k=i}^{i+9}​(h_θ​(x^{(k)})−y^{(k)})x^{(k)}_j​$$
在上面的例子中，我们可以一次归纳10个样本。其相对于一次使用一个样本的优势在于，我们可以用向量化的表述来进行计算。

---

在之前的视频中 我们讨论了随机梯度下降 以及它是怎样比批量梯度下降更快的 在这次视频中 让我们讨论基于这些方法的另一种变形 叫做小批量梯度下降 这种算法有时候甚至比随机梯度下降还要快一点 首先来总结一下我们已经讨论过的算法 在批量梯度下降中每次迭代我们都要用所有的m个样本 然而在随机梯度下降中每次迭代我们只用一个样本 小批量梯度下降做的介于它们之间 准确地说 在这种方法中我们每次迭代使用b个样本 b是一个叫做"小批量规模"的参数 所以这种算法介于随机梯度下降和批量梯度下降之间 这就像批量梯度下降 只不过我会用小很多的批量规模 b的一个标准取值可能是10 比如说 b的一个标准的取值可能是2到100之间的任何一个数 因此那是小批量规模的一个非常典型的取值区间 算法思想是我们每次用b个样本而不是每次用1个或者m个 所以让我正式地把它写出来 我们将要确定b 例如我们假设b是10 所以我们将要从训练集中取出接下来的10个样本 假设训练集是样本 (x(i).y(i)) 的集合 如果是10个样本 最多索引值达到(x(i+9),y(i+9)) 这是全部的10个样本 然后我们将要用这10个样本做一个实际上是梯度下降的更新 因此 就是学习率乘以1/10乘以 k 对 h (x(k)-y(k))×x(k)j 从i到i+9求和 在这个表达式中 我们计算10个样本的梯度下降公式的和 因此 这是数字10 就是小批量规模 i+9 9来自参数b的选择 然后在这之后我们将要把 i 加10 我们将要继续处理接下来的10个样本 然后像这样一直继续 因此完整地写出整个算法 为了简化刚才的这个索引 我将假设我有小批量规模为10和一个大小为1000的训练集 我们接下来要做的就是计算这个形式的和 从i等于1、11、21等等开始 步长是10因为我们每次处理10个样本 然后我们每次对10个样本使用这种梯度下降来更新 所以这是10这是i+9 它们是小批量规模选取10带来的结果 这是最后的for循环 这里在991结束因为 如果我有1000个训练样本那么为了遍历整个训练集我需要100个步长为10的循环 这就是小批量梯度下降 相比批量梯度下降 这种算法也让我们进展快很多 所以让我们再次处理美国3亿人的人口普查数据训练集 然后我们要说的是在处理了前10个样本之后 我们可以开始优化参数θ 因此我们不需要扫描整个训练集 我们只要处理前10个样本然后这可以让我们有所改进 接着我们可以处理第二组10个样本 再次对参数做一点改进 然后接着这样做 因此 这就是小批量梯度下降比批量梯度下降快的原因 你可以在只处理了10个样本之后就改进参数 而不是需要等到你扫描完3亿个样本中的每一个 那么 小批量梯度下降和随机梯度下降比较又怎么样呢？ 也就是说 为什么我们想要每次处理b个样本 而不是像随机梯度下降一样每次处理一个样本？ 答案是——向量化！ 具体来说 小批量梯度下降可能比随机梯度下降好 仅当你有好的向量化实现时 在那种情况下 10个样本求和可以用一种更向量化的方法实现 允许你部分并行计算10个样本的和 因此 换句话说 使用正确的向量化方法计算剩下的项 你有时可以使用好的数值代数库来部分地并行计算b个样本 然而如果你是用随机梯度下降每次只处理一个样本 那么你知道 每次只处理一个样本没有太多的并行计算 至少并行计算更少 小批量梯度下降的一个缺点是有一个额外的参数b 你需要调试小批量大小 因此会需要一些时间 但是如果你有一个好的向量化实现这种方法有时甚至比随机梯度下降更快 好了 这就是小批量梯度下降算法 在某种意义上做的事情介于随机梯度下降和批量梯度下降之间 如果你选择合理的b的值 我经常选择b等于10 但是 你知道 别的值 比如2到100之间的任何一个数都可能合理 因此我们选择b的值 如果你有一个好的向量化实现 有时它可以比随机梯度下降和批量梯度下降更快 



## 17.4、 Stochastic Gradient Descent Convergence(随机梯度下降收敛)

对于随机梯度下降算法，我们该如何选择它的学习率α呢？以及，我们如何确保这个算法能够尽可能的接近全局最优解呢？

一个可行的策略是画出假设模型对于每1000个（或其它数量）训练样本的代价。我们可以在每一轮迭代的时候计算并存储该信息。

![Checking for convergence]()

在这个过程中，使用一个小一点的学习率，有可能会得到一个更接近最优解的方案。这是因为，随机梯度下降算法会在全局最优解附近振荡徘徊，如果使用更小的学习率的话，能够使用振荡步幅更小，进而更接近最优解。

如果你增加用于描绘算法性能的样本平均数量，那么得到的曲线会更加平滑。

如果用于描绘算法性能的样本平均数量太小，会导致曲线的波动较大，更难看出变化趋势。

一个用于逼近全局最优解的策略是，随着时间而缓慢降低学习率。例如，


$$α = \frac{const1}{iterationNumber+const2​}$$


但是实际上人们并不愿意经常这样做，因为这样意味着需要处理更多的参数。

---

现在你已经知道了随机梯度下降算法 但是当你运行这个算法时 你如何确保调试过程已经完成 并且能正常收敛呢？ 还有 同样重要的是 你怎样调整随机梯度下降中学习速率α的值 在这段视频中 我们会谈到一些方法来处理这些问题 确保它能收敛 以及选择合适的学习速率α 回到我们之前批量梯度下降的算法 我们确定梯度下降已经收敛的一个标准方法 是画出最优化的代价函数 关于迭代次数的变化 这就是代价函数 我们要保证这个代价函数在每一次迭代中 都是下降的 当训练集比较小的时候 我们不难完成 因为要计算这个求和是比较方便的 但当你的训练集非常大的时候 你不希望老是定时地暂停算法 来计算一遍这个求和 因为这个求和计算需要考虑整个的训练集 而随机梯度下降的算法是 你每次只考虑一个样本 然后就立刻进步一点点 不需要在算法当中 时不时地扫描一遍全部的训练集 来计算整个训练集的代价函数 因此 对于随机梯度下降算法 为了检查算法是否收敛 我们可以进行下面的工作 让我们沿用之前定义的cost函数 关于θ的cost函数 等于二分之一倍的训练误差的平方和 然后 在随机梯度下降法学习时 在我们对某一个样本进行训练前 在随机梯度下降中 我们要关注样本(x(i),y(i)) 然后关于这个样本更新一小步 进步一点点 然后再转向下一个样本 (x(i+1),y(i+1)) 随机梯度下降就是这样进行的 在算法扫描到样本(x(i),y(i)) 但在更新参数θ之前 使用这个样本 我们可以算出这个样本对应的cost函数 我再换一种方式表达一遍 当随机梯度下降法对训练集进行扫描时 在我们使用某个样本(x(i),y(i))来更新θ前 让我们来计算出 这个假设对这个训练样本的表现 我要在更新θ前来完成这一步 原因是如果我们用这个样本更新θ以后 再让它在这个训练样本上预测 其表现就比实际上要更好了 最后 为了检查随机梯度下降的收敛性 我们要做的是 每1000次迭代 我们可以画出前一步中计算出的cost函数 我们把这些cost函数画出来 并对算法处理的最后1000个样本的cost值求平均值 如果你这样做的话 它会很有效地帮你估计出 你的算法在最后1000个样本上的表现 所以 我们不需要时不时地计算Jtrain 那样的话需要所有的训练样本 随机梯度下降法的这个步骤 只需要在每次更新θ之前进行 也并不需要太大的计算量 要做的就是 每1000次迭代运算中 我们对最后1000个样本的cost值求平均然后画出来 通过观察这些画出来的图 我们就能检查出随机梯度下降是否在收敛 这是几幅画出来的图的例子 假如你已经画出了最后1000组样本的cost函数的平均值 由于它们都只是1000组样本的平均值 因此它们看起来有一点嘈杂 因此cost的值不会在每一个迭代中都下降 假如你得到一种这样的图像 看起来是有噪声的 因为它是在一小部分样本 比如1000组样本中求的平均值 如果你得到像这样的图 那么你应该判断这个算法是在下降的 看起来代价值在下降 然后从大概这个点开始变得平缓 这就是代价函数的大致走向 这基本说明你的学习算法已经收敛了 如果你想试试更小的学习速率 那么你很有可能看到的是 算法的学习变得更慢了 因此代价函数的下降也变慢了 但由于你使用了更小的学习速率 你很有可能会让算法收敛到一个好一点的解 红色的曲线代表随机梯度下降使用一个更小的学习速率 出现这种情况是因为 别忘了 随机梯度下降不是直接收敛到全局最小值 而是在局部最小附近反复振荡 所以使用一个更小的学习速率 最终的振荡就会更小 有时候这一点小的区别可以忽略 但有时候一点小的区别 你就会得到更好一点的参数 接下来再看几种其他的情况 假如你还是运行随机梯度下降 然后对1000组样本取cost函数的平均值 并且画出图像 那么这是另一种可能的图形 看起来这样还是已经收敛了 如果你把这个数 1000 提高到5000组样本 那么可能你会得到一条更平滑的曲线 通过在5000个样本中求平均值 你会得到比刚才1000组样本更平滑的曲线 这是你增大平均的训练样本数的情形 当然增大它的缺点就是 现在每5000个样本才能得到一个数据点 因此你所得到的关于学习算法表现的反馈 就显得有一些“延迟” 因为每5000个样本才能得到图上的一个数据点 而不是每1000个样本就能得到 沿着相似的脉络 有时候你运行梯度下降 可能也会得到这样的图像 如果出现这种情况 你要知道 可能你的代价函数就没有在减小了 也就是说 算法没有很好地学习 因为这看起来一直比较平坦 代价项并没有下降 但同样地 如果你对这种情况时 也用更大量的样本进行平均 你很可能会观察到红线所示的情况 能看得出 实际上代价函数是在下降的 只不过蓝线用来平均的样本数量太小了 并且蓝线太嘈杂 你看不出来代价函数的趋势确实是下降的 所以可能用5000组样本来平均 比用1000组样本来平均 更能看出趋势 当然 即使是使用一个较大的样本数量 比如我们用5000个样本来平均 我用另一种颜色来表示 即使如此 你还是可能会发现 这条学习曲线是这样的 它还是比较平坦 即使你用更多的训练样本 如果是这样的话 那可能就更肯定地说明 不知道出于什么原因 算法确实没怎么学习好  那么你就需要调整学习速率 或者改变特征变量 或者改变其他的什么 最后一种你可能会遇到的情况是 如果你画出曲线 你会发现曲线是这样的 实际上是在上升 这是一个很明显的信号 告诉你算法正在发散 那么你要做的事 就是用一个更小一点的学习速率α 好的 希望通过这几幅图 你能了解到  当你画出cost函数在某个范围的训练样本中求平均值时 各种可能出现的现象 也告诉你 在遇到不同的情况时 应该采取怎样的措施 所以如果曲线看起来噪声较大 或者老是上下振动  那就试试增大你要平均的样本数量 这样应该就能得到比较好的变化趋势 如果你发现代价值在上升 那么就换一个小一点的α值 最后还需要再说一下关于学习速率的问题 我们已经知道 当运行随机梯度下降时 算法会从某个点开始 然后曲折地逼近最小值 但它不会真的收敛 而是一直在最小值附近徘徊 因此你最终得到的参数 实际上只是接近全局最小值 而不是真正的全局最小值 在大多数随机梯度下降法的典型应用中 学习速率α一般是保持不变的 因此你最终得到的结果一般来说是这个样子的 如果你想让随机梯度下降确实收敛到全局最小值 你可以随时间的变化减小学习速率α的值 所以 一种典型的方法来设置α的值 是让α等于某个常数1 除以 迭代次数加某个常数2 迭代次数指的是你运行随机梯度下降的迭代次数 就是你算过的训练样本的数量 常数1和常数2是两个额外的参数 你需要选择一下 才能得到较好的表现 但很多人不愿意用这个办法的原因是 你最后会把问题落实到 把时间花在确定常数1和常数2上 这让算法显得更繁琐 也就是说 为了让算法更好 你要调整更多的参数 但如果你能调整得到比较好的参数的话 你会得到的图形是 你的算法会在最小值附近振荡 但当它越来越靠近最小值的时候 由于你减小了学习速率 因此这个振荡也会越来越小 直到落到几乎靠近全局最小的地方 我想这么说能听懂吧？ 这个公式起作用的原因是 随着算法的运行 迭代次数会越来越大 因此学习速率α会慢慢变小 因此你的每一步就会越来越小 直到最终收敛到全局最小值 所以 如果你慢慢减小α的值到0 你会最后得到一个更好一点的假设 但由于确定这两个常数需要更多的工作量 并且我们通常也对 能够很接近全局最小值的参数 已经很满意了 因此我们很少采用逐渐减小α的值的方法 在随机梯度下降中 你看到更多的还是让α的值为常数 虽然两种做法的人都有 总结一下 这段视频中 我们介绍了一种方法  近似地监测出随机梯度下降算法在最优化代价函数中的表现 这种方法不需要定时地扫描整个训练集 来算出整个样本集的代价函数 而是只需要每次对最后1000个 或者多少个样本 求一下平均值 应用这种方法 你既可以保证随机梯度下降法正在正常运转和收敛 也可以用它来调整学习速率α的大小





## Advanced Topics(进阶主题)
---
## 17.5、 Online Learning(在线学习)

基于一个用户在某网站上的行为而产生的连续数据流，我们能够运行一个无尽的循环来得到（x,y），其中采集的x代表了用户行为的特征，y代表了某种表现。

你可以使用每一对独立的（x,y）来修正模型的参数θ。通过连续不断地更新θ，实现了划分新的用户群体。

![other online learning example]()

---

在这个视频中 我将会 讨论一种新的大规模的 机器学习机制 叫做 在线学习机制 在线学习机制 让我们可以模型化问题 在拥有连续一波数据 或连续的数据流涌进来 而我们又需要 一个算法来从中学习的时候来模型化问题 今天 许多大型网站 或者许多大型网络公司 使用不同版本的 在线学习机制算法 从大批的涌入 又离开网站的用户身上 进行学习 特别要提及的是 如果你有 一个由连续的用户流引发的 连续的数据流 用户流进入 你的网站 你能做的是使用一个 在线学习机制 从数据流中学习 用户的偏好 然后使用这些信息 来优化一些 关于网站的决策 

假定你有一个提供运输服务的公司 所以你知道 用户们来向你询问 把包裹从A地 运到B地的服务 同时假定你有一个网站 让用户们可多次登陆  然后他们告诉你  他们想从哪里寄出包裹 以及 包裹要寄到哪里去 也就是出发地与目的地 然后你的网站开出运输包裹的 的服务价格 比如 我会收取$50来运输你的包裹 我会收取$20之类的 然后根据 你开给用户的这个价格 用户有时会接受这个运输服务 那么这就是个正样本 有时他们会走掉 然后他们拒绝 购买你的运输服务 所以 让我们假定我们想要一个 学习算法来帮助我们 优化我们想给用户 开出的价格 而且特别的是 我们假定 我们找到了一些 获取用户特点的方法 如果我们知道一些用户的统计信息 它们会获取 比如 包裹的起始地 以及目的地 他们想把包裹运到哪里去 以及我们提供给他们的 运送包裹的价格 我们想要做的就是 学习 在给出的价格下他们将会 选择 运输包裹的几率 在已知用户特点的前提下 并且 我要再次指出 他们也同时获取了我们开出的价格 所以如果我们可以 估计出用户选择 使用我们的服务时 我们所开出的价格 那么我们 可以试着去选择 一个优化的价格 因而在这个价格下 用户会有很大的可能性 选择我们的网站 而且同时很有可能会提供给我们 一个合适的回报 让我们 在提供运输服务时也能获得合适的利润 所以如果我们可以学习 y 等于 1 时的条件 在任何给定价格以及其他给定的 条件下y等于1的特征 我们就真的可以利用这一些信息 在新用户来的时候选择合适的价格 所以为了 获得 y 等于 1 的概率的模型 我们能做的就是 用逻辑回归或者神经网络 或者其他一些类似的算法 但现在我们先来考虑逻辑回归 

现在假定你有一个 连续运行的网站 以下就是在线学习算法要做的 我要写下"一直重复" 这只是代表着我们的网站 将会一直继续 保持在线学习 这个网站将要发生的是 一个用户 偶然访问 然后我们将会得到 与其对应的一些(x,y)对 这些(x,y)对是相对应于一个特定的客户或用户的 所以特征 x 是指 客户所指定的起始地与目的地 以及 我们这一次提供 给客户的价格 而y则取1或0 y值取决于 客户是否选择了 使用我们的运输服务 现在我们一旦获得了这个{x,y}数据对 在线学习算法 要做的就是 更新参数θ 利用刚得到的(x,y)数据对来更新θ 具体来说 我们将这样更新我们的参数θ θj 将会被更新为 θj 减去学习率 α 乘以 梯度下降 来做逻辑回归 然后我们对j等于0到n 重复这个步骤 这是我的另一边花括号 所以对于其他的学习算法 不是写(x,y)对 对吧 我之前写的是 (x(i),y(i)) 一样的数据对 但在这个在线学习机制中 我们实际上丢弃了 获取一个固定的数据集这样的概念 取而代之的是 我们拥有一个算法 现在 当我们 获取一个样本 然后我们 利用那个样本获取信息学习 然后我们丢弃这个样本 我们丢弃那个样本 而且我们 永远不会再使用它 这就是为什么我们在一个时间点只会处理一个样本的原因 我们从样本中学习 我们再丢弃它 这也就是为什么 我们放弃了一种拥有 我们放弃了一种拥有 固定的 由 i 来作参数的数据集的表示方法 而且 如果你真的运行 一个大型网站 在这个网站里你有一个连续的 用户流登陆网站 那么 这种在线学习算法 是一种非常合理的算法 因为数据本质上是自由的 如果你有如此多的数据 而数据 本质上是无限的 那么 或许就真的没必要 重复处理 一个样本 当然 如果我们只有 少量的用户 那么我们就不选择像这样的在线学习算法 你可能最好是要 保存好所有的 数据 保存在一个固定的 数据集里 然后对这个数据集使用某种算法 但是 如果你确实有一个连续的 数据流 那么一个 在线学习机制会非常的有效 我也必须要提到一个 这种在线学习算法 会带来的有趣的效果 那就是 它可以对正在变化的用户偏好进行调适 

而且特别的 如果 随着时间变化 因为 大的经济环境发生变化 用户们可能会 开始变得对价格更敏感 然后愿意支付 你知道的 不那么愿意支付高的费用 也有可能他们变得对价格不那么敏感 然后他们愿意支付更高的价格 又或者各种因素 变得对用户的影响更大了 如果你开始拥有 某一种新的类型的用户涌入你的网站 这样的在线学习算法 也可以根据变化着的 用户偏好进行调适 而且从某种程度上可以跟进 变化着的用户群体所愿意 支付的价格 而且 在线学习算法有这样的作用是因为 如果你的用户群变化了 那么参数θ的变化与更新 会逐渐调适到 你最新的用户群所应该体现出来的 参数 这里有另一个 你可能会想要使用在线学习的例子 这是一个对于产品搜索的应用 在这个应用中 我们想要 使用一种学习机制来学习如何 反馈给用户好的搜索列表 举个例子说 你有一个在线 卖电话的商铺 一个卖移动电话或者手机的商铺 而且你有一个用户界面 可以让用户登陆你的网站 并且键入一个 搜索条目 例如“安卓 手机 1080p 摄像头” 那么1080p 是指一个 对应于摄像头的 手机参数 这个参数可以出现在 一部电话中 一个移动电话 或者一个手机中 假定 假定我们的商铺中有一百部电话 而且出于我们的网站设计 当一个用户 键入一个命令 如果这是一个搜索命令 我们会想要找到一个 合适的十部不同手机的列表 来提供给用户 我们想要做的是 拥有一个在线学习机制来帮助我们 找到在这100部手机中 哪十部手机 是我们真正应该反馈给用户的 而且这个返回的列表是对类似这样的用户搜索条目最佳的回应 接下来要说的是一种解决问题的思路 对于每一个手机以及一个给定的 用户搜索命令 我们 可以构建一个 特征矢量x 那么这个特征矢量x 可能会抓取手机的各种特点 它可能会抓取类似于 用户搜索命令与这部电话的类似程度有多高这样的信息 我们获取类似于 这个用户搜索命令中有多少个词 可以与这部手机的名字相匹配 或者这个搜索命令中有多少词 与这部手机的描述相匹配 所以特征矢量x获取 手机的特点而且 它会获取 这部手机与搜索命令 的结果在各个方面的匹配程度 我们想要做的就是 估测一个概率 这个概率是指用户 将会点进 某一个特定的手机的链接 因为我们想要给用户展示 他们 想要买的手机 我们想要给用户提供 那些他们很可能 在浏览器中点进去查看的手机 所以我将定义y等于1时 是指用户点击了 手机的链接 而y等于0是指用户没有点击链接 然后我们想要做的就是 学习到用户 将会点击某一个背给出的特定的手机的概率 你知道的 特征X 获取了手机的特点 以及搜索条目与手机的匹配程度 如果要给这个问题命一个名 用一种运行这类网站的人们 所使用的语言来命名 这类学习问题 这类问题其实被称作 学习预测的点击率 预估点击率CTR 它仅仅代表这学习 用户将点击某一个 特定的 你提供给他们的链接的概率 所以CTR是 点击率(Click Through Rate)的简称 然后 如果你能够估计 任意一个特定手机的点击率 我们可以做的就是 利用这个来 给用户展示十个 他们最有可能点击的手机 因为从这一百个手机中 我们可以计算出 100部手机中 每一部手机的可能的点击率 而且我们选择10部 用户最有可能点击的手机 那么这就是一个非常合理的 来决定要展示给用户的十个搜索结果的方法 更明确地说 假定 每次用户 进行一次搜索 我们回馈给用户十个结果 在线学习算法会做的是 它会真正地提供给我们十个 (x,y) 数据对 这就真的 给了我们十个数据样本 每当一个用户来到 我们网站时就给了我们十个样本 因为对于这十部我们选择 要展示给用户的手机 对于 这10部手机中的每一个 我们会得到 一个特征矢量x 而且 对于这10部手机中的任何一个手机 我们还会得到 y的取值 我们也会观察这些取值 这些取值是根据 用户有没有点击 那个网页链接来决定的 这样 运行此类网站的 一种方法就是 连续给用户展示 你的十个最佳猜测 这十个推荐是指用户可能会喜欢的其他的手机 那么 每次一个用户访问 你将会得到十个 样本 十个(x,y) 数据对 然后利用一个在线学习 算法来更新你的参数 更新过程中会对这十个样本利用10步 梯度下降法 然后 你可以丢弃你的数据了 如果你真的拥有一个连续的 用户流进入 你的网站 这将会是 一个非常合理的学习方法 来学习你的算法中的参数 从而来给用户展示 十部他们 最有可能点击查看的手机 所以 这是一个产品搜索问题 或者说是一个学习将手机排序 的问题 学习搜索手机的样例 接着 我会快速地提及一些其他的例子 其中一个例子是 如果你有 一个网站 你在尝试着 来决定 你知道的 你要给用户 展示什么样的特别优惠 这与手机那个例子非常类似 或者你有一个 网站 然后你想给不同的用户展示不同的新闻文章 那么 如果你是一个新闻抓取网站 那么你又可以 使用一个类似的系统 来选择 来展示给用户 他们最有可能感兴趣的 他们最有可能感兴趣的 新闻文章 以及那些他们最有可能点击的新闻文章 与特别优惠所密切相关的是 我们将会从这些推荐中获利 而且实际上 如果你有 一个协作过滤系统 你可以想象到 一个协作过滤系统 可以给你更多的 特征 这些特征可以整合到 逻辑回归的分类器 从而可以尝试着 预测对于你可能推荐给用户的 不同产品的点击率 当然 我需要说明的是 这些问题中的任何一个都可以 被归类到 标准的 拥有一个固定的样本集的机器学习问题中 或许 你可以运行一个 你自己的网站 尝试运行几天 然后保存一个数据集 一个固定的数据集 然后对其运行 一个学习算法 但是这些是实际的 问题 在这些问题里 你会看到大公司会获取 如此多的数据 所以真的没有必要 来保存一个 固定的数据集 取而代之的是 你可以使用一个在线学习算法来连续的学习 从这些用户不断产生的数据中来学习 

所以 这就是在线学习机制  然后就像我们所看到的 我们所使用的这个算法 与随机梯度下降算法 非常类似 唯一的区别的是 我们不会 使用一个固定的数据集 我们会做的是获取 一个用户样本 从那个样本中学习 然后 丢弃那个样本并继续下去 而且如果你对某一种应用有一个连续的 数据流 这样的算法可能会 非常值得考虑 当然 在线学习的一个优点 就是 如果你有一个变化的 用户群 又或者 你在尝试预测的事情 在缓慢变化 就像你的用户的 品味在缓慢变化 这个在线学习 算法可以慢慢地 调试你所学习到的假设 将其调节更新到最新的 用户行为





## 17.6、 Map Reduce and Data Parallelism (减少映射与数据并行)

我们可以将批处理梯度下降算法的工作量进行切割，将切割出来的子集分割给不同的计算机，这样我们就可以实现对于数据的并行处理了。

你可以将你的训练集按照你拥有的机器数量分割成z个子集。在每一个机器上，计算​$\sum^q_{i=p}​(h_θ​(x^{(i)})−y^{(i)})⋅x^{(i)}_j$​​，其中p和q是每一个子集的分割起始点和终点。

MapReduce将会分发这些任务(map)，然后通过计算来减少它们（reduce）：

For all j=0,1,…,n:

$$Θ_j ​:= Θ_j​−α\frac{1}{z}​(temp^{(1)}_j​+temp^{(2)}_j​+⋯+temp^{(z)}_j​)$$

这样简单地拿到每一台机器计算出的代价，然后计算它们的平均值，乘以学习率，然后更新θ。

总的来说，如果你的算法能够被表达成函数对于训练集的结果之和的形式，那么是可以通过MapReduce来实现的。线性回归、逻辑回归就属于此类。

对于神经网络而言，你可以将数据集化为子集来进行正向传播和反向传播的计算。这些机群最后都会将结果反馈回到master来进行结果的整理。




---

在上面几个视频中 我们讨论了 随机梯度下降 以及梯度下降算法的 其他一些变种 包括如何将其 运用于在线学习 然而所有这些算法 都只能在一台计算机上运行 

但是 有些机器学习问题 太大以至于不可能 只在一台计算机上运行 有时候 它涉及的数据量如此巨大 以至于不论你使用何种算法 你都不希望只使用 一台计算机来处理这些数据 

因此 在这个视频中 我希望介绍 进行大规模机器学习的另一种方法 称为映射约减 (map reduce) 方法 尽管我们 用了多个视频讲解 随机梯度下降算法 而我们将只用少量时间 介绍映射化简 但是请不要根据 我们所花的时间长短 来判断哪一种技术 更加重要 事实上 许多人认为 映射化简方法至少是 同等重要的 还有人认为映射化简方法 甚至比梯度下降方法更重要 我们之所以只在 映射化简上花的时间比较少 只是因为它相对简单 容易解释 然而 实际上 相比于随机梯度下降方法 映射化简方法 能够处理 更大规模的问题 

这个方法如下 假设我们要 拟合一个线性回归模型 或者逻辑回归模型 或者其他的什么模型 让我们再次从随机梯度下降算法开始吧 这就是我们的随机梯度下降学习算法 为了让幻灯片上的文字 更容易理解 我们将假定m固定为400个样本 当然 根据 大规模机器学习的标准 m等于400 实在是太小了 也许在实际问题中 你更有可能遇到 样本大小为4亿 的数据 或者其他差不多的大小 但是 为了使我们的讲解更加简单和清晰 我们假定我们只有400个样本 这样一来 随机梯度下降学习算法中 这里是400 以及400个样本的求和 这里i从1取到400 如果m很大 那么这一步的计算量将会很大 

因此 下面我们来介绍 映射化简算法 这里我必须指出 映射化简算法的基本思想 来自Jeffrey Dean和Sanjay Ghemawat 这两位研究者 Jeff Dean是硅谷 最为传奇般的 一位工程师 今天谷歌 (Google) 所有的服务 所依赖的后台基础架构 有很大一部分是他创建的 

扯远了 我们还是回到映射化简的基本思想 假设我们有一个 训练样本 我们将它表示为 这个方框中的一系列x~y数据对 

从x(1) y(1)开始 涵盖我所有的400个样本 直到x(m) y(m) 总之 这就是我的400个训练样本 

根据映射化简的思想 一种解决方案是 将训练集划分成几个不同的子集 

好 我们基本上已经讲完了两本书 我假定我有 4台计算机 它们并行的 处理我的训练数据 因此我要将数据划分成4份 分给这4台计算机 如果你有10台计算机 或者100台计算机 那么你可能会将训练数据划分成10份或者100份 

我的4台计算机中 第一台 将处理第一个 四分之一训练数据 也就是前100个训练样本

具体来说 这台计算机 将参与处理这个求和 它将对前100个训练样本进行求和运算 

让我把公式写下来吧 我将计算临时变量 

temp(1) 这里的上标(1) 表示第一台计算机 

其下标为j 该变量等于从1到100的求和 然后我在这里写的部分 和这里的完全相同 也就是h_θ(x(i))-y(i) 

乘以x(i)j 这其实就是 这里的梯度下降公式中的这一项 然后 类似地 我将用第二台计算机 处理我的 第二个四分之一数据 也就是说 我的第二台计算机 将使用第101到200号训练样本 类似地 我们用它 计算临时变量 temp(2)_j 也就是从101到200号 数据的求和 类似的 第三台和第四台 计算机将会使用 第三个和第四个 四分之一训练样本 这样 现在每台计算机 不用处理400个样本 而只用处理100个样本 它们只用完成 四分之一的工作量 这样 也许可以将运算速度提高到原来的四倍 

最后 当这些计算机 全都完成了各自的工作 我会将这些临时变量 

收集到一起 我会将它们 送到一个 中心计算服务器 这台服务器会 将这些临时变量合并起来 具体来说 它将根据以下公式 来更新参数θj 新的θj将等于 

旧的θj减去 学习速率α乘以 400分之一 乘以临时变量 temp(1)_j 加temp(2)_j 加temp(3)_j 

加temp(4)_j 当然 对于j等于0的情况我们需要单独处理 这里 j从0 取到特征总数n 

通过将这个公式拆成多行讲解 我希望大家已经理解了 其实 这个公式计算的数值 

和原先的梯度下降公式计算的数值 是完全一样的 只不过 现在我们有一个中心运算服务器 它收集了一些部分计算结果 temp(1)_j temp(2)_j temp(3)_j 和 temp(4)_j 把它们加了起来 很显然 这四个 临时变量的和 

就是这个求和 加上这个求和 再加上这个求和 再加上这个求和 它们加起来的和 其实和原先 我们使用批量梯度下降公式 计算的结果是一样的 

接下来 我们有 α乘以400分之一 这里也是α乘以400分之一 因此这个公式 完全等同于批量梯度下降公式 唯一的不同是 我们原本需要在一台计算机上 完成400个训练样本的求和 而现在 我们将这个工作分给了4台计算机 

总结来说 映射约减技术是这么工作的 

我们有一些训练样本 如果我们希望使用4台计算机 并行的运行机器学习算法 那么我们将训练样本等分 尽量均匀地分成4份 

然后 我们将这4个 训练样本的子集送给4台不同的计算机 每一台计算机 对四分之一的 训练数据 进行求和运算 最后 这4个求和结果 被送到一台中心计算服务器 负责对结果进行汇总 在前一张幻灯片中 在那个例子中 梯度下降计算 的内容是对i等于1到400的 400个样本进行求和运算 更宽泛的来讲 在梯度下降计算中 我们是对i等于1到m的m个样本 进行求和 现在 因为这4台计算机 的每一台都可以 完成四分之一的计算工作 因此你可能会得到4倍的加速 

特别的 如果没有网络延时 也不考虑 通过网络来回传输数据 所消耗的时间 那么你可能可以得到4倍的加速 当然 在实际工作中 因为网络延时 数据汇总额外消耗时间 以及其他的一些因素 你能得到的加速总是略小于4倍的 但是 不管怎么说 这种映射化简算法 确实让我们能够处理 将能够处理 所无法处理的大规模数据 

如果你打算 将映射化简技术用于 加速某个机器学习算法 也就是说 你打算运用多台不同的计算机 并行的进行计算 那么你需要问自己一个很关键的问题 那就是 你的机器学习算法 是否可以表示为训练样本的某种求和 事实证明 很多机器学习算法 的确可以表示为 关于训练样本的函数求和 而在处理大数据时 这些算法的主要运算量 在于对大量训练数据求和 因此 只要你的机器学习算法 可以表示为 训练样本的一个求和 只要算法的 主要计算部分 可以表示为 训练样本的求和 那么你可以考虑使用映射化简技术 

来将你的算法扩展到非常大规模的数据上 

让我们再看一个例子 

假设我们想使用某种高级优化算法 比如说 LBFGS算法 或者共轭梯度算法等等 假设我们想使用逻辑回归算法 于是 我们需要计算两个值 对于LBFGS算法和共轭梯度算法 我们需要计算的第一个值是 我们需要提供一种方法 用于计算 优化目标的代价函数值 

比如 对于逻辑回归 你应该记得它的代价函数 可以表示为 训练样本上的这种求和 因此 如果你想在 10台计算机上并行计算 那么你需要将训练样本 分给这10台计算机 让每台计算机 计算10份之一 训练数据的 

是异常困难的 高级优化算法 还需要提供 这些偏导数 的计算方法 同样的 对于逻辑回归 这些偏导数 可以表示为 训练数据的求和 因此 和之前的例子类似 你可以让 每台计算机只计算 

部分训练数据上的求和 

最后 当这些求和计算完成之后 求和结果 会被发送到 一台中心计算服务器上 这台服务器将对结果进行再次求和 这等同于 对临时变量temp(i) 或者 temp(i)_j 进行求和 而这些临时标量 是第i台计算机算出来的 中心计算服务器 对这些临时变量求和 得到了总的代价函数值 以及总的偏导数值 然后你可以将这两个值传给高级优化函数 

因此 更广义的来说 通过将机器学习算法 表示为 求和的形式 或者是 训练数据的函数求和形式 你就可以运用映射化简技术 来将算法并行化 这样就可以处理大规模数据了 

最后再提醒一点 目前我们只讨论了 运用映射化简技术 在多台计算机上 实现并行计算 也许是一个计算机集群 也许是一个数据中心中的多台计算机 

但实际上 有时即使我们只有一台计算机 

我们也可以运用这个技术

具体来说 现在的许多计算机 都是多核的 你可以有多个CPU 而每个CPU 又包括多个核 如果你有一个 很大的训练样本 那么你可以 使用一台 四核的计算机 即使在这样一台计算机上 你依然可以 将训练样本分成几份 然后让每一个核 处理其中一份子样本 这样 在单台计算机 或者单个服务器上 你也可以利用映射化简技术来划分计算任务 每一个核 可以处理 比方说四分之一 训练样本的求和 然后我们再将 这些部分和汇总 最终得到整个训练样本上的求和 相对于多台计算机 这样在单台计算机上 使用映射化简技术 的一个优势 在于 现在你不需要 担心网络延时问题 因为所有的通讯 所有的来回数据传输 

都发生在一台计算机上 因此 相比于使用数据中心的 多台计算机 现在网络延时的影响 小了许多 最后 关于在一台多核计算机上的并行运算 我再提醒一点 这取决于你的编程实现细节 如果你有一台 多核计算机 并且使用了某个线性代数函数库 

那么请注意 某些线性代数函数库 

会自动利用多个核 并行地完成线性代数运算 

因此 如果你幸运地 使用了这种 线性代数函数库 当然 并不是每个函数库都会自动并行 但如果你用了这样一个函数库 并且你有一个矢量化得很好的算法实现 

那么 有时你只需要 按照标准的矢量化方式 实现机器学习算法 而不用管多核并行的问题 

因为你的线性代数函数库会自动帮助你完成多核并行的工作 因此 这时你不需要使用映射化简技术 但是 对于其他的问题 使用基于映射化简的实现 寻找并使用 适合映射化简的问题表述 然后实现一个 多核并行的算法 可能是个好主意 它将会加速你的机器学习算法 

在这个视频中 我们介绍了映射化简(map reduce)技术 它可以通过 将数据分配到多台计算机的方式 来并行化机器学习算法 实际上这种方法 也可以利用 单台计算机的多个核 

广告 今天 网上有许多优秀的 开源映射化简实现 实际上 一个称为Hadoop 的开源系统 已经拥有了众多的用户 通过自己实现映射化简算法 或者使用别人的开源实现 你就可以利用映射化简技术 来并行化机器学习算法 这样你的算法 将能够处理 单台计算机处理不了的大数据 




### Review