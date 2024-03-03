from __future__ import absolute_import

import os
import tensorflow as tf
import numpy as np
import random
import math

def conv2d(inputs, filters, strides, padding):
	"""
	Performs 2D convolution given 4D inputs and filter Tensors.
	:param inputs: tensor with shape [num_examples, in_height, in_width, in_channels]
	:param filters: tensor with shape [filter_height, filter_width, in_channels, out_channels]
	:param strides: MUST BE [1, 1, 1, 1] - list of strides, with each stride corresponding to each dimension in input
	:param padding: either "SAME" or "VALID", capitalization matters
	:return: outputs, Tensor with shape [num_examples, output_height, output_width, output_channels]
	"""
	num_examples = inputs.shape[0] # tk: None? 
	in_height = inputs.shape[1]
	in_width = inputs.shape[2]
	input_in_channels = inputs.shape[3]

	filter_height = filters.shape[0]
	filter_width = filters.shape[1]
	filter_in_channels = filters.shape[2]
	filter_out_channels = filters.shape[3]

	num_examples_stride = strides[0]
	strideY = strides[1]
	strideX = strides[2]
	channels_stride = strides[3]

	strides = [num_examples_stride, strideY, strideX, channels_stride]
	# inputs.shape = [num_examples, in_height, in_width, input_in_channels]
	# filter.shape = [filter_height, filter_width, filter_in_channels, filter_out_channels]
	
	assert(input_in_channels == filter_in_channels), "Number of input channels must match number of filter channels"

	if strides != [1, 1, 1, 1]:
		raise ValueError("Strides must be [1, 1, 1, 1]")
	
	# Cleaning padding input
	if padding == "SAME": # tk else? how to use padding
		padY = (filter_height - 1)/2 #height
		padX = (filter_width - 1)/2
		# x_pad = np.pad(x, ((0,0), (2, 2), (2, 2), (0,0)), mode='constant', constant_values = (0,0))
	else: 
		padY = 0
		padX = 0
	
	inputs = np.pad(inputs,  ((0,0), (math.floor(padY), math.floor(padY)), (math.floor(padX), math.floor(padX)), (0,0)), mode='constant', constant_values = (0,0))  #tk 

	# Calculate output dimensions
	output_height = int((in_height + 2*padY - filter_height) / strideY + 1)
	# xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)

	# kern - filter; input - Img

	output_width = int((in_width + 2*padX - filter_width) / strideX + 1)	

	#  input = [num_examples, in_height, in_width, in_channels]
	#  input_data = np.random.random((4, 100, 3, 2)) -> (4, 100, 2, 16)
	output = np.zeros((num_examples, output_height, output_width, filter_out_channels))
	
	print(f"my conv2d: {tf.shape(output)}")
	print(f"tf.nn.conved: {tf.nn.conv2d(inputs, filters, strides, padding)}")
	print(f"tf.nn.conved: {tf.shape(tf.nn.conv2d(inputs, filters, strides, padding))}")


	for b in range(num_examples):
		for h in range(output_height): 
			for w in range(output_width):
				for i in range(input_in_channels):
					for o in range(filter_out_channels):
						output[b, h, w, o] \
						+= np.sum(filters[:, :,i, o] * inputs[b, h: h + filter_height, w: w + filter_width, i])

	# output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
	print(f"print my conv2d again: {tf.shape(output)}")
	print(f"my conv2d: {output}")



	# PLEASE RETURN A TENSOR. HINT: 
	output = tf.convert_to_tensor(output, dtype = tf.float32)

	return output
def same_test_0():
	'''
	Simple test using SAME padding to check out differences between 
	own convolution function and TensorFlow's convolution function.

	NOTE: DO NOT EDIT
	'''
	imgs = np.array([[2,2,3,3,3],[0,1,3,0,3],[2,3,0,1,3],[3,3,2,1,2],[3,3,0,2,3]], dtype=np.float32)
	imgs = np.reshape(imgs, (1,5,5,1))
	filters = tf.Variable(tf.random.truncated_normal([2, 2, 1, 1],
								dtype=tf.float32,
								stddev=1e-1),
								name="filters")
	my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="SAME")
	tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="SAME")
	print("SAME_TEST_0:", "my conv2d:", my_conv[0][0][0], "tf conv2d:", tf_conv[0][0][0].numpy())

def valid_test_0():
	'''
	Simple test using VALID padding to check out differences between 
	own convolution function and TensorFlow's convolution function.

	NOTE: DO NOT EDIT
	'''
	imgs = np.array([[2,2,3,3,3],[0,1,3,0,3],[2,3,0,1,3],[3,3,2,1,2],[3,3,0,2,3]], dtype=np.float32)
	imgs = np.reshape(imgs, (1,5,5,1))
	filters = tf.Variable(tf.random.truncated_normal([2, 2, 1, 1],
								dtype=tf.float32,
								stddev=1e-1),
								name="filters")
	my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="VALID")
	tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="VALID")
	print("VALID_TEST_0:", "my conv2d:", my_conv[0][0], "tf conv2d:", tf_conv[0][0].numpy())

def valid_test_1():
	'''
	Simple test using VALID padding to check out differences between 
	own convolution function and TensorFlow's convolution function.

	NOTE: DO NOT EDIT
	'''
	imgs = np.array([[3,5,3,3],[5,1,4,5],[2,5,0,1],[3,3,2,1]], dtype=np.float32)
	imgs = np.reshape(imgs, (1,4,4,1))
	filters = tf.Variable(tf.random.truncated_normal([3, 3, 1, 1],
								dtype=tf.float32,
								stddev=1e-1),
								name="filters")
	my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="VALID")
	tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="VALID")
	print("VALID_TEST_1:", "my conv2d:", my_conv[0][0], "tf conv2d:", tf_conv[0][0].numpy())

def valid_test_2():
	'''
	Simple test using VALID padding to check out differences between 
	own convolution function and TensorFlow's convolution function.

	NOTE: DO NOT EDIT
	'''
	imgs = np.array([[1,3,2,1],[1,3,3,1],[2,1,1,3],[3,2,3,3]], dtype=np.float32)
	imgs = np.reshape(imgs, (1,4,4,1))
	filters = np.array([[1,2,3],[0,1,0],[2,1,2]]).reshape((3,3,1,1)).astype(np.float32)
	my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="VALID")
	tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="VALID")
	print("VALID_TEST_1:", "my conv2d:", my_conv[0][0], "tf conv2d:", tf_conv[0][0].numpy())

def my_same_test_1():
	'''
	Simple test using SAME padding to check out differences between 
	own convolution function and TensorFlow's convolution function.
	'''
	imgs = np.array([[2,2,3,3,3],[0,1,3,0,3],[2,3,0,1,3],[3,3,2,1,2],[3,3,0,2,3]], dtype=np.float32)
	imgs = np.reshape(imgs, (1,5,5,1))
	filters = tf.Variable(tf.random.truncated_normal([2, 2, 1, 1],
								dtype=tf.float32,
								stddev=1e-1),
								name="filters")
	my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="SAME")
	tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="SAME")
	print("SAME_TEST_0:", "my conv2d:", my_conv[0][0][0].numpy(), "tf conv2d:", tf_conv[0][0][0].numpy())



def main():
	# TODO: Add in any tests you may want to use to view the differences between your and TensorFlow's output
	# same_test_0()
	# valid_test_0()
	valid_test_2()


if __name__ == '__main__':
	main()
