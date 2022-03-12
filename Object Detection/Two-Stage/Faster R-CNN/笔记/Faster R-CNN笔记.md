# Faster R-CNN笔记

[TOC]

## $\sect0$ 背景

R-CNN 使目标检测的效果得到了巨大的提升，但是该模型处理图像速度较慢，Fast R-CNN 等改进模型增快处理速度并提升性能，region proposal的生成（通过selective search等）成为速度与性能的瓶颈。

## $\sect1$ 主要创新点

1. 使用 RPN 进行region proposal的生成。
2. RPN 与 Fast R-CNN 共享CNN特征提取模块。
3. 使用 Anchor 取代特征金字塔、卷积金字塔等结构。

## $\sect2$ Faster R-CNN 结构

由两部分组成：$\Big\{^{获取region proposal的卷积网络RPN}_{Fast\space R-CNN探测头}$

### $\sect2.0$ RPN

一个 fully convolutional network。

1. 首先通过一个backbone网络 (论文中使用VGG 或 ZFNet)对图像特征进行提取获得其特征映射。该部分与探测头共用。

2. 使用一个滑动窗口网络在 (1) 中得到的特征映射上进行滑动，每次将 $n\times n$ 的特征映射部分输入，将其映射至更低维度的特征映射。此后将该特征映射同时输入两个平行的网络：

   1. 框回归模块：回归得到候选框的四个参数(x, y, w, h)。
   2. 框分类模块：将候选框进行分类 (具体目标分类与背景分类)。

   以上两个子网络使用线性全连接网络。为了实现这种结构，可以使用$n\times n$ 卷积与 $1\times1$ 卷积的串联加以实现。在这个部分使用了anchor结构，具体相关信息见 $\sect3$。

### $\sect2.1$ 探测头

沿用了Fast R-CNN 的配置(RoI池化等)。

### $\sect2.2$ 共用特征提取模块

Faster R-CNN 的两个模块(RPN 和探测头)中都需要对图像特征进行提取，因此为了提高性能与加速算法，两模块共享用于提取特征映射的CNN模块。如下图，最下部分为特征提取CNN模块，输入图片经过该模块产生特征映射。然后经过左部分 RPN 结构产生 region proposals，再将region proposals与特征映射经过 RoI 池化与分类得到最终结果 (Fast R-CNN 中的架构)。

<img src="img\structure.jpg" style="zoom:50%;" />

## $\sect3$ Anchors

在$\sect2.0$中的滑窗网络部分，采用了anchor结构同时预测多个region proposals。其思想在于通过改变预选框的尺寸大小(例如$256^2$，$512^2$等)以及高宽比(例如1:1, 1:2等)同时至多获取k($=n_{scale}\times n_{aspect\space ratio}$) 个预选区域。因此一个滑窗网络的输出为$k(n_{reg}+n_{cls})=(4+c)k$个，其中c为分类总数。值得一提的是anchor结构具有平移不变性，即目标的空间位置发生变化不影响其预测结果。这使得该模型相较于当时的一些其他模型可以减小模型尺寸并防止过拟合。

## $\sect4$ 模型训练

### $\sect4.0$ RPN的损失函数

简化模型中的分类为二分类(前景vs背景)，则
$$
L(\{p_i\}, \{t_i\})=\frac{1}{N_{cls}}\sum_i{L_{cls}(p_i,p_i^*)}+\frac{\lambda}{N_{reg}}\sum_i{p_i^*L_{reg}(t_i,t_i^*)}
$$
其中星号代表真实标签与框，不带星号为预测值，$L_{cls}$ 为对数损失，$L_{reg}$为rebust loss。乘以$p^*$意味着只有正类损失有效，背景不考虑框回归。$N_{cls}$在只有两类时由batch大小决定，$N_{reg}$由anchor数决定。平衡系数$\lambda$ 默认取10。注意当anchor的IoU大于阈值(例如0.7)或为IoU最大则为正类，IoU小于阈值(例如0.15)则标记为背景类，其余anchor舍弃。

### $\sect4.1$ 边界框回归

与 Fast R-CNN 不同，后者的框回归进行于RoI 池化产生的特征映射，而前者中发生于RPN中。考虑到不同尺寸，对每种anchor各训练一个回归器。

每个回归器需要训练四个参数：
$$
\begin{align}
t_x&=(x-x_a)/w_a\\
t_y&=(y-y_a)/h_a\\
t_w&=\log(w/w_a)\\
t_h&=\log(h/h_a)\\
\end{align}
$$
根据实际标注，目标值为
$$
\begin{align}
t_x^*&=(x^*-x_a)/w_a\\
t_y^*&=(y^*-y_a)/h_a\\
t_w^*&=\log(w^*/w_a)\\
t_h^*&=\log(h^*/h_a)\\
\end{align}
$$
据此进行回归。

### $\sect4.2$ 采样问题

由于背景类样本数过多的样本不均衡，直接使用会使模型偏向于背景类，本文中采用image-metric采样策略，从一个图片获得的batch中随机采样256个anchor，其中正负类之比不大于1，除非正类数不足一个batch的一半，此时使用负类进行填充。

### $\sect4.3$ 四步交替训练

1. 使用如前所述的方法训练RPN，模型初始化为ImageNet预训练模型。
2. 利用RPN产生的region proposal单独训练探测头，该模型亦初始化为ImageNet预训练模型，此时两部分并不共享特征提取；
3. 固定并共享卷积特征提取，使用探测头初始化RPN训练，只调整RPN独有部分（滑窗+并行的分类回归网络）；
4. 固定并共享卷积特征提取，训练探测头独有部分。

## $\sect5$ 一些细节

- NMS(非极大值抑制)不会影响mAP但可以降低假警报。

- cls分数对最高排名一部分目标准确率影响较大。

