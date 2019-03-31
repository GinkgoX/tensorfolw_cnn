#alexnet application for image sets identification
#source http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#needed file:	bvlc_alexnet.npy -- the weights; they need to be in the working directory
#				caffe_classes.py -- the classes, in the same order as the outputs of the network

import tensorflow as tf
import numpy as np

#define convLayer to generate different conv layer
def convLayer(x, k_height, k_width, stride_x, stride_y, feature_num, name, padding = 'SAME', groups = 1):
	'''
	descrption: to generate different conv layer
	Args:		x: input data [image or last conv result]
				k_height: kernal height
				k_width: kernal width
				stride_x: strides x step
				stride_y: stride y step
				feature_num: numbers of feature [kernal size]
				name: tf string name
				padding: padding mode [default: 'SAME']
				groups: the architecture of alexnet [default groups = 1 (1 GPU)]
	Returns:	relu(x*w + b)
	'''
	#access the image channal
	channal = int(x.get_shape()[-1])
	#define conv anonymous function
	conv = lambda image, kernal : tf.nn.conv2d(image, kernal, strides = [1, stride_x, stride_y, 1], padding = padding)
	#define name variable for weights and biases
	with tf.variable_scope(name) as scope:
		w = tf.get_variable('w', shape = [k_height, k_width, channal / groups, feature_num])
		b = tf.get_variable('b', shape = [feature_num])
		#divide the tensor as sub_tensor and reoutput x_ and w_
		x_ = tf.split(value = x, num_or_size_splits = groups, axis = 3)
		w_ = tf.split(value = w, num_or_size_splits = groups, axis = 3)
		#extract feature map for x and w
		feature_map = [conv(v_x, v_w) for v_x, v_w in zip(x_, w_)]
		#merge feature map
		merge_feature_map = tf.concat(axis = 3, values = feature_map)
		#calcuate output
		out = tf.nn.bias_add(merge_feature_map, b)
		#relu activation
		return tf.nn.relu(tf.reshape(out, merge_feature_map.get_shape().as_list()), name = scope.name)

#define fcLayer function to generate different full conncetion layer
def fcLayer(x, in_dim, out_dim, relu_flag, name):
	'''
	descrption:	to generate different full conncetion layer
	Args:	x: the input data
			in_dim: input dimension
			out_dim: output dimension
			relu_flag: use the relu activation or not
			name : tf name string
	Returns: relu activated result
	'''
	with tf.variable_scope(name) as scope:
		#get name variable w and b
		w = tf.get_variable('w', shape = [in_dim, out_dim], dtype = 'float')
		b = tf.get_variable('b', shape = [out_dim], dtype = 'float')
		#calcuate (x*w + b)
		out = tf.nn.xw_plus_b(x, w, b, name = scope.name)
		if relu_flag:
			return tf.nn.relu(out)
		return out

#define alexnet model class to set up alexnet network
class alexnet(object):
	'''
	descrption: to set up alexnet network
	function:	_init_ -- set parameters and initialize the network
				nn -- the cnn architecture
				load -- to load model
	'''
	#define _init_ function to initialize the parameters for alexnet model
	def __init__(self, x, keep_pro, class_num, model_path = 'bvlc_alexnet.npy'):
		'''
		descrption: to initialize parameters for alexnet model
		Args:		x: input data[image]
					keep_pro: keep alive probility of neruial units
					class_num: numbers of class
					model_path: the trained model file
		'''
		self.X = x
		self.KEEPPRO = keep_pro
		self.CLASSNUM = class_num
		self.MODELPATH = model_path
		self.nn()

	#define nn function to set up the alexnet network
	def nn(self):
		'''
		descrption: to set oup the alexnet network
		Args: 	self parameters
		Returns: 	None
		'''
		#conv 1 layer
		conv_1 = convLayer(self.X, 11, 11, 4, 4, 96, 'conv1', 'VALID')
		#max pooling layer
		pool_1 = tf.nn.max_pool(conv_1, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'VALID', name = 'pool1')
		#lrn operation
		lrn_1 = tf.nn.lrn(pool_1, depth_radius = 2, alpha = 2e-05, beta = 0.75, bias = 1.0, name = 'norm1')
		#conv 2 layer
		conv_2 = convLayer(lrn_1, 5, 5, 1, 1, 256, 'conv2', groups = 2)
		#max pooling layer
		pool_2 = tf.nn.max_pool(conv_2, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'VALID', name = 'pool2')
		#lrn operation
		lrn_2 = tf.nn.lrn(pool_2, depth_radius = 2, alpha = 2e-05, beta = 0.75, bias = 1.0, name = 'lrn2')
		#conv 3 layer
		conv_3 = convLayer(lrn_2, 3, 3, 1, 1, 384, 'conv3')
		#conv 4 layer
		conv_4 = convLayer(conv_3, 3, 3, 1, 1, 384, 'conv4', groups = 2)
		#conv 5 layer
		conv_5 = convLayer(conv_4, 3, 3, 1, 1, 256, 'conv5', groups = 2)
		#max pooling layer
		pool_5 = tf.nn.max_pool(conv_5, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'VALID', name = 'pool5')
		#full conncetion 1 layer
		fc_in = tf.reshape(pool_5, [-1, 256*6*6])
		fc_1 = fcLayer(fc_in, 256*6*6, 4096, True, 'fc6')
		#dropout operation
		dropout_1 = tf.nn.dropout(fc_1, self.KEEPPRO)
		#full conncetion 2 layer
		fc_2 = fcLayer(dropout_1, 4096, 4096, True, 'fc7')
		#dropout operation
		dropout_2 = tf.nn.dropout(fc_2, self.KEEPPRO)
		#full conncetion 3 layer
		self.fc_3 = fcLayer(dropout_2, 4096, self.CLASSNUM, True, 'fc8')
	#define load function to load alexnet model
	def load(self, sess):
		'''
		descrption: to load alexnet model
		Args:	sess: tf session
		Returns:	None
		'''
		file_dict = np.load(self.MODELPATH, encoding = 'bytes').item()
		for name in file_dict:
			if name not in []:
				with tf.variable_scope(name, reuse = True):
					for parameter in file_dict[name]:
						if len(parameter.shape) == 1:
							sess.run(tf.get_variable('b', trainable = False).assign(parameter))
						else:
							sess.run(tf.get_variable('w', trainable = False).assign(parameter))

#test model
import os
import cv2
import caffe_classes

if __name__ == '__main__':
	#set parameters
	keep_pro = 1
	class_num = 1000
	test_path = "testimage"
	#read test image
	testImg = []
	for i in os.listdir(test_path):
		testImg.append(cv2.imread(test_path + '/' + i))

	img_mean = np.array([104, 117, 124], np.float)
	
	#define input placeholder
	x = tf.placeholder('float', [1, 227, 227, 3])
	
	#load alexnet model
	model = alexnet(x, keep_pro, class_num)
	
	#display score
	score = model.fc_3
	print(score)
	
	#define softmax variable
	softmax = tf.nn.softmax(score)
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		#load model
		model.load(sess)
		for i, img in enumerate(testImg):
			#resize the image and substract mean value
			test = cv2.resize(img.astype(np.float), (227, 227)) - img_mean
			#convert image to test
			test = test.reshape((1, 227, 227, 3))
			#take index where the predict accuracy is most
			max_index = np.argmax(sess.run(softmax, feed_dict = {x: test}))
			#take the most probility class
			res = caffe_classes.class_names[max_index]
			print(res)
			#set font
			font = cv2.FONT_HERSHEY_SIMPLEX
			#display class name
			cv2.putText(img, res, (int(img.shape[0] / 3), int(img.shape[1] / 3)), font, 1, (0, 0, 255), 2)
			#show result
			cv2.imshow('test', img)
			cv2.waitKey(0)
