# VGG

## Article

Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition[J]. arXiv preprint arXiv:1409.1556, 2014.

## Aim

To evaluate networks of increasing depth using an architecture with very small ($3\times3$) convolution filters.

## Results

1. Representation depth is beneficial for the classification accuracy.
2. LRN doesn't help with increasing accuracy.
3. Additional nonlinearity helps.
4. scale jittering helps.
5. Multi-scale helps.
6. Multi-crop better than dense, but the fusion is even better.
7. Multi-crop evaluation is complementary to a dense evaluation.

## Methods

1. Use small conv filters:
   1. add more nonlinearity and be more discriminative;
   2. decrease the number of parameters.

2. $1\times1$ conv layers: increase nonlinearity but not affect receptive fields.
3. To circumvent initialization problem, pre-train a shallow network and then insert conv layers with random initialization.
4. augmentation by single-scale training and multi-scale training (scale jittering): rescale the image $\to$ crop to the training scale; rescale randomly(can be of different sizes) $\to$ crop. 

