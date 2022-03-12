# AlexNet

## Article

Krizhevsky A, Sutskever I, Hinton G E. Imagenet classification with deep convolutional neural networks[J]. Advances in neural information processing systems, 2012, 25: 1097-1105.

## Aim

To train a large, deep convolutional neural network to classify images in the ImageNet LSVRC\-2010 contest.

## Result

The model achieved error rates which is considerably better than the previous SOTA.

## Methods

Architecture:

1. Use ReLU nonlinearity which is not saturating. It accelerate the training and make it possible to train a deep CNN.
2. Train on multiple GPUs for the lack of memory.
3. Use local response normalization to aid generalization.
4. Use overlapping pooling(max pooling) to reduce overfit.

Reducing overfitting

1. Data augmentation: extract tinier-size patches from images and their horizonal reflections; alter the intensity of the RGB channels with PCA.
2. Dropout: actually sample a different architecture sharing weights to force to reduce co-adaptations, but doubles iterations to converge.

Optimizer: SGD with momentum.

## comments

1. Use of CNN, ReLU, GPU greatly changed the zone of image recognition.
2. Some methods like LRN are later demonstrated to be not so useful.