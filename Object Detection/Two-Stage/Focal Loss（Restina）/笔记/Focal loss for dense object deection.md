# Focal loss for dense object detection笔记

[TOC]

## $\sect0$ 背景

目标检测主要思路有one-stage与two-stage两种，其中two-stage方法先提取出候选区域，这些候选区域数量较为稀疏，再在其上进行目标检测任务准确率较高但速度较慢。而one-stage方法在密集的区域上进行检测任务，速度较快但准确率低。作者认为这种现象的原因在于在密集区域上进行检测会出现目标/背景样本不平衡。因此提出了focal loss来解决该问题并给出RestinaNet模型。

## $\sect1$ Focal loss

对于二分类任务，令$p_t=\big\{^{p\space\space\space\space\space\space if\space y=1}_{1-p\space if\space y=0}$，则常见的交叉熵损失函数为
$$
CE(p_t)=-\log p_t
$$
当正负类样本比例不均衡时，少类的识别效果会很差，而目标检测任务中恰恰需要检测的目标数量远小于背景数量，因此one-stage方法的准确率难以提升。为了避免该问题，为每一类赋予一个权重系数，构建$\alpha$-平衡交叉熵：
$$
CE^\prime(p)=-\alpha_t \log p_t
$$
同时由于背景类大部分较为容易分类，即其$p_t$较大，对于特征学习贡献不大且降低模型泛化能力，需要将关注点更多放在困难、模糊的实例的分类，因此加入参数$(1-p_t)^\gamma$，得到focal loss的表达式：
$$
FL(p_t)=-\alpha_t(1-p_t)^\gamma \log p_t
$$

## $\sect2$ RestinaNet

使用FPN 作为backbone，两个全卷积网络分别进行对象分类和边框回归。

## $\sect3$ 实验结果

Focal loss在不影响正类的判断情况下，背景类大幅减少。

$\gamma$取2结果较好。

将$\gamma$ 变大时$\alpha$ 应缩小。