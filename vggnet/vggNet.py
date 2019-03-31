#vggnet network model
from datetime import datetime
import math
import time
import tensorflow as tf

#define batch_size and num_batches
batch_size = 32
num_batches = 100

#define conv_op to generate different conv layer
def conv_op(x, k_height, k_width, stride_x, stride_y, chan_num, name, para_list):
	'''
	description: to generate different conv layer
	Args:	x: the input data
			k_height: the kernal height
			k_width: the kernal width
			stride_x: strides x step
			stride_y: strides y step
			chan_num: channel numbers
			name: tf name string
			para_list: parameter list [kernal, biases]
	Returns:	activation: the activation result of relu(x*w +b)
	'''
	chan_in = x.get_shape()[-1].value
	with tf.name_scope(name) as scope:
		#get kernal variable
		kernal = tf.get_variable(scope + 'w', shape = [k_height, k_width, chan_in, chan_num], 
								dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer_conv2d())
		#define conv operation
		conv = tf.nn.conv2d(x, kernal, strides = [1, stride_x, stride_y, 1], padding = 'SAME')
		#initialize bias value
		bias = tf.constant(0.0, shape = [chan_num], dtype = tf.float32)
		biases = tf.Variable(bias, trainable = True, name = 'b')
		#calculate the result with biases
		res = tf.nn.bias_add(conv, biases)
		#calculate the relu result
		activation = tf.nn.relu(res, name = scope)
		#calculate parameter list
		para_list += [kernal, biases]
		return activation

#define fc_op to generate different full connection layer
def fc_op(x, chan_num, name, para_list):
	'''
	description: to generate different full connection layer
	Args:	x: the input data
			chan_num : the output channel numbers
			name : tf name string
			para_list : the parameter list
	Returns:	activation: the activation result of relu(x*w + b)
	'''
	chan_in = x.get_shape()[-1].value
	with tf.name_scope(name) as scope:
		#get kernal variable
		kernal = tf.get_variable(scope + 'w', shape = [chan_in, chan_num],
								dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer_conv2d())
		#define biases
		biases = tf.Variable(tf.constant(0.1, shape = [chan_num], dtype = tf.float32), name = 'b')
		#relu operation
		activation = tf.nn.relu_layer(x, kernal, biases, name = scope)
		#calculate parameter list
		para_list += [kernal, biases]
		return activation

#define pool_op to excute max_pool operation
def pool_op(x, k_height, k_width, stride_x, stride_y, name):
	'''
	description: to excute max_pool operation
	Args:	x: the input data
			k_height: kernal height
			k_width: kernal width
			stride_x: strides x step
			stride_y: strides y step
			name: tf name string
	Returns: result of max_pool operation
	'''
	return tf.nn.max_pool(x, ksize = [1, k_height, k_width, 1], strides = [1, stride_x, stride_y, 1], padding = 'SAME', name = name)

#define inference_op to build the vgg_16 network
def inference_op(x, keep_pro):
	'''
	description: to build the vgg_16 network
	Args:	x: the input data
			keep_pro: the kept neurial probility of dropout operation
	Returns:	prediction: the prediction class
				softmax: the softmax activation result
				fc8: full connection layer 8 result
				para_list: parameter list
	'''
	para_list = []
	
	#define first part conv
	conv1_1 = conv_op(x, k_height = 3, k_width = 3, stride_x = 1, stride_y = 1,chan_num = 64, name = 'conv1_1', para_list = para_list)
	conv1_2 = conv_op(conv1_1, k_height = 3, k_width = 3, stride_x = 1, stride_y = 1, chan_num = 64, name = 'conv1_2', para_list = para_list)
	pool_1 = pool_op(conv1_2, k_height = 2, k_width = 2, stride_x = 2, stride_y = 2, name = 'pool_1')
	
	#define second part conv
	conv2_1 = conv_op(pool_1, k_height = 3, k_width = 3, stride_x = 1, stride_y = 1, chan_num = 128, name = 'conv2_1', para_list = para_list)
	conv2_2 = conv_op(conv2_1, k_height = 3, k_width = 3, stride_x = 1, stride_y = 1, chan_num = 128, name = 'conv2_2', para_list = para_list)
	pool_2 = pool_op(conv2_2, k_height = 2, k_width = 2, stride_x = 2, stride_y = 2, name = 'pool_2')
	
	#define third part conv
	conv3_1 = conv_op(pool_2, k_height = 3, k_width = 3, stride_x = 1, stride_y = 1, chan_num = 256, name = 'conv3_1', para_list = para_list)
	conv3_2 = conv_op(conv3_1, k_height = 3, k_width = 3, stride_x = 1, stride_y = 1, chan_num = 256, name = 'conv3_2', para_list = para_list)
	conv3_3 = conv_op(conv3_2, k_height = 3, k_width = 3, stride_x = 1, stride_y = 1, chan_num = 256, name = 'conv3_3', para_list = para_list)
	pool_3 = pool_op(conv3_3, k_height = 2, k_width = 2, stride_x = 2, stride_y = 2, name = 'pool_3')
	
	#define fourth part conv
	conv4_1 = conv_op(pool_3, k_height = 3, k_width = 3, stride_x = 1, stride_y = 1, chan_num = 512, name = 'conv4_1', para_list = para_list)
	conv4_2 = conv_op(conv4_1, k_height = 3, k_width = 3, stride_x = 1, stride_y = 1, chan_num = 512, name = 'conv4_2', para_list = para_list)
	conv4_3 = conv_op(conv4_2, k_height = 3, k_width = 3, stride_x = 1, stride_y = 1, chan_num = 512, name = 'conv4_3', para_list = para_list)
	pool_4 = pool_op(conv4_3, k_height = 2, k_width = 2, stride_x = 2, stride_y = 2, name = 'pool_4')
	
	#define fivth part conv
	conv5_1 = conv_op(pool_4, k_height = 3, k_width = 3, stride_x = 1, stride_y = 1, chan_num = 512, name = 'conv5_1', para_list = para_list)
	conv5_2 = conv_op(conv5_1, k_height = 3, k_width = 3, stride_x = 1, stride_y = 1, chan_num = 512, name = 'conv5_2', para_list = para_list)
	conv5_3 = conv_op(conv5_2, k_height = 3, k_width = 3, stride_x = 1, stride_y = 1, chan_num = 512, name = 'conv5_3', para_list = para_list)
	pool_5 = pool_op(conv5_3, k_height = 2, k_width = 2, stride_x = 2, stride_y = 2, name = 'pool_5')
	
	#reshape pool_5
	sp = pool_5.get_shape()
	flatten = tf.reshape(pool_5, [-1, sp[1].value * sp[2].value * sp[3].value], name = 'flatten')
	
	#define full connection layer 6
	fc6 = fc_op(flatten, chan_num = 4096, name = 'fc6', para_list = para_list)
	fc6_drop = tf.nn.dropout(fc6, keep_pro, name = 'fc6_drop')
	
	#define full connection layer 7
	fc7 = fc_op(fc6_drop, chan_num = 4096, name = 'fc7', para_list = para_list)
	fc7_drop = tf.nn.dropout(fc7, keep_pro, name = 'fc7_drop')
	
	#define full connection layer 8
	fc8 = fc_op(fc7_drop, chan_num = 1000, name = 'fc8', para_list = para_list)
	
	#get softmax results
	softmax = tf.nn.softmax(fc8)
	
	#get the prediction class results
	prediction = tf.argmax(softmax, 1)
	return prediction, softmax, fc8, para_list

#define time_tensorflow_run function to access the time consuming
def time_tensorflow_run(sess, operator,feed, test_name):
	'''
	description: to access the time consuming
	Args:	sess: tensorflow Session
			operator: operator for accessing
			feed: the feed_dict input
			test_name: test name
	Returns:	time consuming info
	'''
	#define pre hot batch size
	num_steps_burn_in = 10
	#define total duration
	total_duration = 0.0
	#define total duration square error
	total_duration_squared = 0.0
	
	#calculate time consuming and print result every 10 epoch
	for i in range(num_batches + num_steps_burn_in):
		start_time = time.time()
		_ = sess.run(operator, feed_dict = feed)
		duration = time.time() - start_time
		if i >= num_steps_burn_in:
			if not i % 10 :
				print('%s:	step %d, 	duration := %.3f'%(datetime.now(), i - num_steps_burn_in, duration))
			total_duration += duration
			total_duration_squared = duration * duration
	#calculate avarage time consuming
	ave_time = total_duration / num_batches
	#calculate strandand deviance
	var = ave_time*ave_time - total_duration_squared / num_batches
	std_dev = math.sqrt(var)
	print('%s,	%s across %d steps, %.3f, +/- %.3f sec/batch'%(datetime.now(), test_name, num_batches, ave_time, std_dev))

#define run_benchmark function to create session and test model
def run_benchmark():
	with tf.Graph().as_default():
		#define image size
		image_size = 224
		#create random image pixel
		images = tf.Variable(tf.truncated_normal([batch_size, image_size, image_size, 3], dtype = tf.float32, stddev = 0.1))
		#define keep_pro value
		keep_pro = tf.placeholder(tf.float32)
		#call the inference_op to get the info
		prediction, softmax, fc8, para_list = inference_op(images, keep_pro)
		#initialize variables
		init = tf.global_variables_initializer()
		sess = tf.Session()
		sess.run(init)
		#call time consuming
		time_tensorflow_run(sess, prediction, {keep_pro:1.0}, 'Forward')
		obj = tf.nn.l2_loss(fc8)
		grad = tf.gradients(obj, para_list)
		time_tensorflow_run(sess, grad, {keep_pro:0.5}, 'Farward-backward')

#run the graph
run_benchmark()
