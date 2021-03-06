# SENet 笔记

## 一、背景及基本思路

一些方法通过强化空间编码能力提升了网络的表示能力，比如Inception增加深度、BN规范化中间层输入等，有一些方法关注到通道间关系，但主要目标通常是缩小模型或减少计算复杂度。同时Attention的思想流行，可以对各通道施加某种注意力机制增强模型表示能力。

SENet思路在于关注通道间关系，通过显式对通道间的依赖性进行建模来提升CNN效果，再根据通道间关系进行特征的重新采样实现通道注意力机制。具体地，对于上一层的特征输出，对每一个通道通过全局平均池化获取一个通道的内容描述（squeeze），再根据这个内容描述对各个通道进行重新分配权重，从而按照权重重新校正输出上一层生成的特征（excitation）。

这种方法是对原本模块的改造思路，可以直接使用原本模块的对应SE版本直接取代原模块（感觉适合用装饰器实现？）。其在模型不同层级的作用不同：在低层激活类别无关的表达能力强的特征，改善了低层次的表达能力；在高层则对类别非常敏感。

## 二、SE 模块

SE的想法非常直接：既然同一层输出的各个通道表示能力、相关性不同，那么直接使用单个通道的一个全局内容的描述来分配各个通道的权重，从而对各个通道根据表示能力进行重新调整进行输出来增加表示能力。在这里全局内容的表示使用了一个简单的全局平均池化（感觉过于简单了，后续有很多改进的思路），根据全局内容描述生成通道权重使用一个两层神经网络。

总结来说，对于上一层输出的各层输出$\{F_i\}$，基本想法可以用下面的式子进行表达：
$$
\begin{align}
D_i&=GAP(F_i)\\
W_i&=NN(D_i)\\
Ouput_i&=W_i\cdot F_i
\end{align}
$$
上式中的GAP为全局平均池化，NN为一个两层神经网络（激活分别为ReLU以及Sigmoid，采用瓶颈型结构）。

## 三、进行模块的SE转换

非残差的模块直接在模块最后接入SE处理即可，对于残差连接在进行残差处理和处理结果与原输出相加之前插入SE处理。

## 四、复杂度讨论

压缩部分只增加了极少的计算，激活部分增加复杂度较多，但依旧只增加了不到0.5%的计算。参数量增加量约10%，而且其中很大一部分在模型高层部分，这部分对性能的提升效果较小可以去除。因此SE方法在增加极少运算和参数的情况下大幅提升了模型效果。