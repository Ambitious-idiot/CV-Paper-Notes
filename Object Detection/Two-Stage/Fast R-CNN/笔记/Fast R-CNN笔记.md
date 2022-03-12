# Fast R-CNN笔记

[TOC]

## $\sect0$ 背景

当时的目标检测 SOTA 主要是以下两个模型架构：R-CNN 和 SPP-net。

R-CNN 主要缺陷：

- 需要多个训练环节；
- 训练对空间和时间耗费太多
- 测试时处理流程太慢 (原因在于对每个region proposal分别进行CNN特征提取没有进行共享)。

SPP-net 对 R-CNN 进行了改进，通过共享特征映射来进行加速，但依旧存在缺陷：

- 多环节训练以及训练的时空高耗费依旧存在；
- 特征提取层的CNN难以通过SPP反向传播的梯度在detection阶段进行有效优化 (主要原因在于RoI感受野过大，池化返回梯度大部分为零无法有效反向传播进行优化)。

## $\sect1$ 主要改进点

1. 将训练改进为一个单阶段流程；
2. 所有模块在训练过程中可以得到有效优化；
3. 训练过程无硬盘存储。

## $\sect2$ 整体架构

输入图片经过CNN提取特征得到特征映射，经过Selective search得到region proposals，两者依据位置关系相结合得到RoIs，每一个RoI经过RoI pooling得到压缩的特征映射并经过全连接层得到特征向量，根据特征向量由两个并联的全连接层分别得到各分类得分((k+1)个结果，每一个类及背景类)、各分类的边界框(4k个结果)。（这里总感觉有点不对劲，因为仅根据RoI内部形成的特征向量去进行边界框回归似乎对RoI周围特征映射理应有所利用，不知道Faster R-CNN中对框回归前移至region proposal stage是不是这个原因。）

流程如下图：
$$
img
^{\stackrel{CNN}{\longrightarrow}Feature\space Map}
_{\stackrel{SS}{\longrightarrow}Region\space Proposals}
\stackrel{+}{\longrightarrow}RoIs
\stackrel{RoI\space pooling}{\longrightarrow}
\stackrel{FC}{\longrightarrow}Feature\space vectors
^{\stackrel{FC}{\longrightarrow}cls}
_{\stackrel{FC}{\longrightarrow}bbox}
$$

## $\sect3$ RoI Pooling

设置H和W两个超参数，这两个超参数要保证与其后的全连接层输入通道数保持统一。将$w\times h$ 的特征映射输入经过$\frac{w}{W}\times\frac{h}{H}$的最大池化保证与全连接层的输入统一。实际可以理解为简化的SPP，即只有一层金字塔的SPP。

## $\sect4$ 迁移ImageNet预训练模型

模型使用VGG16 的backbone，将最后一次max pooling替换成RoI pooling，将用于分类的全连接+softmax替换为两个并联线性任务，同时输入转换为图片和RoIs（原文为RoI，但我感觉region proposals或许更加贴切合理。）

## $\sect5$ Multi-task loss

本模型进行端对端的训练流程，其损失函数为一个多任务损失，即对于回归任务与分类任务采取两种不同的损失函数并进行加权组合：
$$
L=L_{cls}(p,cls)+\lambda\bold1_{cls\gt0}L_{reg}(t^{cls},t^*)
$$
其中$L_{cls}(p,cls)=-\log p_{cls}$，$L_{reg}{t^{cls},t^*}=\sum_{i\in\{x,y,w,h\}}{smooth_{L1}(t_i^{cls}-t^*_i)}$, 其中
$$
\begin{align}
smooth_{L1}(x)&=\Big\{
^{0.5x^2\space\space\space\space\space |x|\lt1}
_{|x|-0.5\space\space otherwise}
\end{align}
$$
其中smooth L1损失函数也叫robust 损失函数，其对于L2损失的优势体现在对离群值不过分敏感。

值得注意的是，此处的分类cls中0表示背景类，其不进行框回归。IoU大于阈值(0.5)为正类，小于阈值(0.1)为负类，正类负类比例保持1:3.两阈值之间使用hard example mining(原文中后两种情况的判定与此处笔记相反，我认为应该是作者笔误)。

## $\sect6$ 一些细节

1. 使用NMS；
2. 使用SVD压缩全连接层；
3. 实验发现过多的region proposals会导致准确率降低。(此后的Focal loss论文中认为原因在于过多的背景负类导致了样本不平衡进而导致该现象出现)