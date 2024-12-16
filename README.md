# dl-cnn

CNN for CIFAR2 (Cats vs. Dogs)

## Results

Implemented a Convolutional Neural Network (CNN) to perform binary classification on the CIFAR dataset,  distinguishing between cats and dogs (CIFAR2). 

Test Accurarcy = 0.756600022315979
Training Accuracy = 0.7523091435432434

## Bugs
One of my major bugs was in `train()`, specifically in my batch processing that resulted in an infinite loop and timing out. 

## Dataset

- Input: CIFAR2 (subset of CIFAR10) images.
- Output: Binary classification (cat = 0, dog = 1).

## Features 
- Custom Convolution Function: Implements a manually written conv2d() for testing
- Visualization: Includes methods to visualize loss and model predictions.

## Steps

1. Preprocess data to normalize pixel values and convert labels to one-hot encoding.
2. Train the model using TensorFlow and evaluate accuracy.
3. Test with the custom `conv2d()` function to validate manual convolution implementation.

## Handout 

https://hackmd.io/@BrownDeepLearningS24/hw3p