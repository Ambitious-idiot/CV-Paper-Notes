# Deep Image Prior 笔记

## 一、基本思路

在计算机视觉中有一类问题是对给定的退化图片进行恢复，比如图像去噪、超分辨率、图像恢复等。对于这些低层次的基础任务，作者想要使用 CNN 直接提取图像先验信息，在**不需要大量训练集进行学习**的情况下直接在**单一的退化图片**上得到恢复的图像输出。这种方法也构建起了基于学习的神经网络方法和手工设计的统计方法之间的联系。

由于卷积神经网络的网络结构，CNN 即使没有训练也天生具有获取图片统计信息的能力。具体地，CNN可以较快地学习图片的天然的低频的信息部分而对高频噪声具有天然的阻抗性。因此，可以直接使用一个未经训练的CNN，在单张退化图片上进行前向计算，再利用输出的生成结果与退化图片本身的相似性作为损失函数，优化神经网络的权重参数。这样就可以通过控制训练轮数，使网络恰好学习了天然信息而没有来得及学习噪音部分时结束训练从而获得较好的生成结果。

对以上过程做数学化的描述，对于以下问题：
$$
给定退化图片\space x_0,\space生成目标\space x^\star=min_x(E(x,x_0)+R(x))
$$
式中 E 为与任务相关的损失函数，R 为正则项。由于神经网络本身可以获取先验知识，所以可以使用神经网络提供了**隐含的正则项**，所以上问题转化为
$$
\begin{align}
给定退化图片\space x_0,\space求解\space\theta^\star&=argmin_\theta E(f_\theta(z),x_0)\\
x^\star&=f_{\theta^\star}(z)
\end{align}
$$
式中$f_\theta(z)$ 为在以$\theta$为权重参数的卷积神经网络上，输入随机噪音$z$获得的生成输出。该转化问题即为本文的研究目标。

## 二、应用

### 1. 图像去噪

在去噪问题中，损失函数为
$$
E(x,x_0)=||x-x_0||^2
$$
通过选择合适的优化器以及合适的训练轮数，即可得到去噪的输出。注意当训练轮数过少，输出尚未充分学习到结构知识，而训练轮数过多会在噪音图像上过拟合而无法达到去噪效果。

### 2. 超分辨率

在超分问题中，损失函数为
$$
E(x,x_0) = ||d(x)-x_0||^2
$$
式中d为降采样。通过将上采样的结果降采样计算与原图的相似程度作为损失。由于本文方法无法获得高分辨率图片的实际性态，该方法的效果无法与基于学习的方法比如基于GAN的方法相媲美，但是显著优于其他手工提取方法。

### 3. 图像修复

在修复问题中，损失函数为
$$
E(x,x_0)=||(x-x_0)|_{完整部分}||^2
$$
即仅使用图像中完整的部分计算相似性，残缺部分不计入相似计算。这种方法修复效果较好但难以应对复杂语义场景，这些场景下使用其他基于encoder\-decoder的方法效果更好。

## 三、总结

本文提供了一种生成恢复方面的新方法，不基于学习而是直接利用生成结果与原图的相似作为量度使用单张退化图片优化生成网络参数，从而利用卷积神经网络天生的先验统计知识获取能力得到生成结果。但是网络的选取方法仍需进一步考虑，训练轮次的控制也须进一步考虑，训练时间过长也需要改进。

