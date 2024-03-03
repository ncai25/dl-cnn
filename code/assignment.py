from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data
from convolution import conv2d

import os
import tensorflow as tf
import numpy as np
import random
import math

# ensures that we run only on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
def create_variable(*dims, initializer=tf.random.truncated_normal):
    ## *dims takes all unnamed variables and condenses to dims list
    return tf.Variable(initializer(dims, stddev=.1))

class Model(tf.keras.Model):
    def __init__(self):
        """
        This model class will contain the architecture for your CNN that 
        classifies images. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Model, self).__init__()

        #Hyperparameter initialization
        self.epochs = 10
        self.batch_size = 64 # tk 500
        self.num_classes = 2
        self.loss_list = [] # Append losses to this list in training so you can visualize loss vs time in main

        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001) # tk

        self.conv1_filter = create_variable(5,5,3,16)
        # 5 height, 5 width, 3 input channels, 16 output channels
        # 16 5x5 filters per channel
        self.conv1_bias = create_variable(16) # tf.zeros([16]) #tk

        self.conv2_filter = create_variable(5,5,16,20) 
        self.conv2_bias = create_variable(20)

        self.conv3_filter = create_variable(3,3,20,20) 
        self.conv3_bias = create_variable(20)

        self.dense1_weight = create_variable(4 * 4 * 20, 128) #tk
        self.dense1_bias = create_variable(128)
        self.dense2_weight = create_variable(128, 64)
        self.dense2_bias = create_variable(64)
        self.dense3_weight = create_variable(64, 2)
        self.dense3_bias = create_variable(2) # eventually 2 classes 


    def call(self, inputs, is_testing=False):
        """
        Runs a forward pass on an input batch of images.
        
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
        # shape of filter = (filter_height, filter_width, in_channels, out_channels)
        # shape of strides = (batch_stride, height_stride, width_stride, channels_stride)


        conv1 = tf.nn.conv2d(inputs, self.conv1_filter, strides=[1, 2, 2, 1], padding="SAME")
        conv1 = tf.nn.bias_add(conv1, self.conv1_bias)
        # mean1, var1 = tf.nn.moments(conv1, axes=[0, 1, 2])
        # conv1 = tf.nn.batch_normalization(conv1, mean1, var1, offset=None, scale=None, variance_epsilon=1e-6)
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.nn.max_pool(conv1, [1, 3, 3, 1],[1, 2, 2, 1], padding="SAME") # batch size + num channels 

        conv2 = tf.nn.conv2d(conv1, self.conv2_filter, strides=[1, 1, 1, 1], padding="SAME")
        conv2 = tf.nn.bias_add(conv2, self.conv2_bias)
        # mean2, var2 = tf.nn.moments(conv2, axes=[0, 1, 2])
        # conv2 = tf.nn.batch_normalization(conv2, mean2, var2, offset=None, scale=None, variance_epsilon=1e-6)
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.nn.max_pool(conv2, [1, 2, 2, 1],[1, 2, 2, 1], padding="SAME") # batch size + num channels 

        if is_testing == False:
            conv3 = tf.nn.conv2d(conv2, self.conv3_filter, strides=[1, 1, 1, 1], padding="SAME")
        else: 
            conv3 = conv2d(conv2, self.conv3_filter, strides=[1, 1, 1, 1], padding="SAME")    

        conv3 = tf.nn.bias_add(conv3, self.conv3_bias)
        # mean3, var3 = tf.nn.moments(conv3, axes=[0, 1, 2])
        # conv3 = tf.nn.batch_normalization(conv3, mean3, var3, offset=None, scale=None, variance_epsilon=1e-6)
        conv3 = tf.nn.relu(conv3)
        # print(conv3)

        # conv3 = tf.reshape(conv3, [-1, 8 * 8])
        conv3 = tf.reshape(conv3, [-1, 4 * 4 * 20])
        # l3_out = tf.reshape(l3_out, [l3_out.shape[0], -1]) ## <- Incorrect way bc batch

        dense1 = tf.nn.relu(tf.matmul(conv3, self.dense1_weight) + self.dense1_bias) 
        dense1 = tf.nn.dropout(dense1, 0.3)

        dense2 = tf.nn.relu(tf.matmul(dense1, self.dense2_weight) + self.dense2_bias) 
        dense2 = tf.nn.dropout(dense2, 0.3)  

        # dense3 = tf.nn.softmax(tf.matmul(dense2, self.dense3_weight) + self.dense3_bias) 
        # print(dense3)
        # negative -> 0 bc of relu 
        dense3 = (tf.matmul(dense2, self.dense3_weight) + self.dense3_bias)

        return dense3 # logits


    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Softmax is applied in this function.
        
        :param logits: during training, a matrix of shape (batch_size, self.num_classes) 
        containing the result of multiple convolution and feed forward layers
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """

        losses = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
        return tf.reduce_mean(losses)

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.
        
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

        NOTE: DO NOT EDIT
        
        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs 
    and labels - ensure that they are shuffled in the same order using tf.gather or zipping.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.
    
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training), 
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training), 
    shape (num_labels, num_classes)y across batches of the train inputs/labels
    '''

    batch_size = model.batch_size
    accuracies = []

    indices = tf.random.shuffle(range(0, train_inputs.shape[0])) # range - not just len
    shuffled_inputs = tf.gather(train_inputs, indices) 
    shuffled_labels = tf.gather(train_labels, indices)

    for b, b1 in enumerate(range(batch_size, shuffled_inputs.shape[0] + 1, batch_size), 1):
        b0 = b1 - batch_size
        batch_inputs = shuffled_inputs[b0:b1]
        batch_inputs = tf.image.random_flip_left_right(batch_inputs)
        batch_labels = shuffled_labels[b0:b1]

        with tf.GradientTape() as tape:
            logits = model.call(batch_inputs) # y_pred
            loss = model.loss(logits, batch_labels)
            model.loss_list.append(loss)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        batch_accuracy = model.accuracy(logits, batch_labels)
        accuracies.append(batch_accuracy)

    return np.mean(accuracies)


def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly 
    flip images or do any extra preprocessing.
    
    :param test_inputs: test data (all images to be tested), 
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """

    batch_size = model.batch_size
    accuracies = []

    for b, b1 in enumerate(range(batch_size, test_inputs.shape[0] + 1), batch_size):
        b0 = b1 - batch_size
        batch_inputs = test_inputs[b0:b1]
        batch_labels = test_labels[b0:b1]
        pred_labels = model.call(batch_inputs)

        accuracies.append(model.accuracy(pred_labels, batch_labels))
    
    return np.mean(accuracies)


def visualize_loss(losses): 
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()  


def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
    """
    Uses Matplotlib to visualize the correct and incorrect results of our model.
    :param image_inputs: image data from get_data(), limited to 50 images, shape (50, 32, 32, 3)
    :param probabilities: the output of model.call(), shape (50, num_classes)
    :param image_labels: the labels from get_data(), shape (50, num_classes)
    :param first_label: the name of the first class, "cat"
    :param second_label: the name of the second class, "dog"

    NOTE: DO NOT EDIT

    :return: doesn't return anything, two plots should pop-up, one for correct results,
    one for incorrect results
    """
    # Helper function to plot images into 10 columns
    def plotter(image_indices, label): 
        nc = 10
        nr = math.ceil(len(image_indices) / 10)
        fig = plt.figure()
        fig.suptitle("{} Examples\nPL = Predicted Label\nAL = Actual Label".format(label))
        for i in range(len(image_indices)):
            ind = image_indices[i]
            ax = fig.add_subplot(nr, nc, i+1)
            ax.imshow(image_inputs[ind], cmap="Greys")
            pl = first_label if predicted_labels[ind] == 0.0 else second_label
            al = first_label if np.argmax(
                image_labels[ind], axis=0) == 0 else second_label
            ax.set(title="PL: {}\nAL: {}".format(pl, al))
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
        
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = image_inputs.shape[0]

    # Separate correct and incorrect images
    correct = []
    incorrect = []
    for i in range(num_images): 
        if predicted_labels[i] == np.argmax(image_labels[i], axis=0): 
            correct.append(i)
        else: 
            incorrect.append(i)

    plotter(correct, 'Correct')
    plotter(incorrect, 'Incorrect')
    plt.show()


def main():
    '''
    Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and 
    test your model for a number of epochs. We recommend that you train for
    10 epochs and at most 25 epochs.

    Consider printing the loss, training accuracy, and testing accuracy after each epoch
    to ensure the model is training correctly.
    
    CS1470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=70%.
    
    CS2470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=75%.
    
    :return: None
    '''
    # TODO: Use the autograder filepaths to get data before submitting to autograder.
    #       Use the local filepaths when running on your local machine.
    AUTOGRADER_TRAIN_FILE = '../data/train'
    AUTOGRADER_TEST_FILE = '../data/test'

    LOCAL_TRAIN_FILE = '/Users/noracai/Documents/CS1470/homework-3p-cnns-norafk-1/data/train'
    LOCAL_TEST_FILE = '/Users/noracai/Documents/CS1470/homework-3p-cnns-norafk-1/data/test'

    # train_inputs, train_labels = get_data(LOCAL_TRAIN_FILE, 3, 5)
    # test_inputs, test_labels = get_data(LOCAL_TEST_FILE, 3, 5)

    train_inputs, train_labels = get_data(AUTOGRADER_TRAIN_FILE, 3, 5)
    test_inputs, test_labels = get_data(AUTOGRADER_TEST_FILE, 3, 5)

    model = Model()

    for e in range(model.epochs):
        train_accuracy = train(model, train_inputs, train_labels)
        print(f"Epoch {e + 1}/{model.epochs}, Training Accuracy: {train_accuracy}")

    test_accuracy = test(model, test_inputs, test_labels)
    print(f"Test Accuracy: {test_accuracy}")

if __name__ == '__main__':
    main()