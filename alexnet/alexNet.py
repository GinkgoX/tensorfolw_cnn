from datetime import datetime
import tensorflow as tf
import math
import time

#define batch_size and 
batch_size = 32
num_batches = 100

#define print_activations function to print each layer's info
def print_activations(tensor):
	'''
	description: to print each layer's info
	Args:	tensor: each input tensor
	Returns:	tensor's name and shape
	'''
	print(tensor.op.name, '', tensor.get_shape().as_list())

#define variable_weights_with_l2_loss function to initialize weights value with l2 regularation
	def variable_weights_with_l2_loss(shape, stddev, w):
		'''
		description: to initialize weights value with l2 regularation
		Args:	shape: the input data shape
				stddev: strandand deviance
				w: weights coffient
		Returns:	weights
		'''
		weights = tf.Variable(tf.Variable(shape, stddev = stddev))
		if w is not None:
			weight_loss = tf.multiply(tf.nn.l2_loss(weights), w, name = 'weight_loss')
			tf.add_to_collection('losses', weight_loss)
		return weights

#define inference function to build alexnet model
def inference(image):
	'''
	description: to build alexnet model
	Args:	image: the input image
	Returns:	fc_3: full connection layer 3 [output layer]
				parameters: total layer's parameters
	'''
	
	#initialize parameters dict
	parameters = []
	
	#define conv_1 layer with max_pool
	with tf.name_scope('conv_1') as scope:
		#define conv kernal
		kernal = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype = tf.float32, stddev = 0.1), name = 'weights')
		#define conv op
		conv = tf.nn.conv2d(image, kernal, [1, 4, 4, 1], padding = 'SAME')
		#define biases
		biases = tf.Variable(tf.constant(0.0, shape = [64], dtype = tf.float32), trainable = True, name = 'biases')
		#linear op for(x*weights + biases)
		bias = tf.nn.bias_add(conv, biases)
		#relu activation
		conv_1 = tf.nn.relu(bias, name = scope)
		#print conv_1 layer info and parameters
		print_activations(conv_1)
		#calculate total parameters
		parameters += [kernal, biases]
		#define lrn_1 op [optional]
		lrn_1 = tf.nn.lrn(conv_1, depth_radius = 4, alpha = 0.001 / 9, beta = 0.75, name = 'lrn_1')
		#define pool_1 op
		pool_1 = tf.nn.max_pool(conv_1, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'VALID', name = 'pool_1')
		#print pool_1 layer info
		print_activations(pool_1)
	
	#define conv_2 layer with max_pool
	with tf.name_scope('conv_2') as scope:
		#define conv kernal
		kernal = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype = tf.float32, stddev = 0.1), name = 'weights')
		#define conv op
		conv = tf.nn.conv2d(pool_1, kernal, [1, 1, 1, 1], padding = 'SAME')
		#define biases
		biases = tf.Variable(tf.constant(0.0, shape = [192], dtype = tf.float32), trainable = True, name = 'biases')
		#linear op for(x*weights + biases)
		bias = tf.nn.bias_add(conv, biases)
		#relu activation
		conv_2 = tf.nn.relu(bias, name = scope)
		#print conv_2 layer info and parameters
		print_activations(conv_2)
		#calculate total parameters
		parameters += [kernal, biases]
		#define lrn_2 op [optional]
		lrn_2 = tf.nn.lrn(conv_2, depth_radius = 4, alpha = 0.001 / 9, beta = 0.75, name = 'lrn_2')
		#define pool_2 op
		pool_2 = tf.nn.max_pool(conv_2, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'VALID', name = 'pool_2')
		#print pool_2 layer info
		print_activations(pool_2)
	
	#define conv_3 layer without max_pool
	with tf.name_scope('conv_3') as scope:
		#define conv kernal
		kernal = tf.Variable(tf.truncated_normal([3, 3, 192, 384], dtype = tf.float32, stddev = 0.1), name = 'weights')
		#define conv op
		conv = tf.nn.conv2d(pool_2, kernal, [1, 1, 1, 1], padding = 'SAME')
		#define biases
		biases = tf.Variable(tf.constant(0.0, shape = [384], dtype = tf.float32), trainable = True, name = 'biases')
		#linear op for(x*weights + biases)
		bias = tf.nn.bias_add(conv, biases)
		#relu activation
		conv_3 = tf.nn.relu(bias, name = scope)
		#print conv_3 layer info and parameters
		print_activations(conv_3)
		#calculate total parameters
		parameters += [kernal, biases]
	
	#define conv_4 layer without max_pool
	with tf.name_scope('conv_4') as scope:
		#define conv kernal
		kernal = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype = tf.float32, stddev = 0.1), name = 'weights')
		#define conv op
		conv = tf.nn.conv2d(conv_3, kernal, [1, 1, 1, 1], padding = 'SAME')
		#define biases
		biases = tf.Variable(tf.constant(0.0, shape = [256], dtype = tf.float32), trainable = True, name = 'biases')
		#linear op for(x*weights + biases)
		bias = tf.nn.bias_add(conv, biases)
		#relu activation
		conv_4 = tf.nn.relu(bias, name = scope)
		#print conv_4 layer info and parameters
		print_activations(conv_4)
		#calculate total parameters
		parameters += [kernal, biases]
	
	#define conv_5 layer with max_pool
	with tf.name_scope('conv_5') as scope:
		#define conv kernal
		kernal = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype = tf.float32, stddev = 0.1), name = 'weights')
		#define conv op
		conv = tf.nn.conv2d(conv_4, kernal, [1, 1, 1, 1], padding = 'SAME')
		#define biases
		biases = tf.Variable(tf.constant(0.0, shape = [256], dtype = tf.float32), trainable = True, name = 'biases')
		#linear op for(x*weights + biases)
		bias = tf.nn.bias_add(conv, biases)
		#relu activation
		conv_5 = tf.nn.relu(bias, name = scope)
		#print conv_5 layer info and parameters
		print_activations(conv_5)
		#calculate total parameters
		parameters += [kernal, biases]
		#define pool_5 op
		pool_5 = tf.nn.max_pool(conv_5, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'VALID', name = 'pool_5')
		print_activations(pool_5)
	return pool_5, parameters

#define time_tensorflow_run function to access the time consuming
def time_tensorflow_run(sess, operator, test_name):
	'''
	description: to access the time consuming
	Args:	sess: tensorflow Session
			operator: operator for accessing
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
		_ = sess.run(operator)
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
		#calculate fc_3 and parameters
		pool_5, parameters = inference(images)
		#initialize variables
		init = tf.global_variables_initializer()
		sess = tf.Session()
		sess.run(init)
		
		#call time consuming
		time_tensorflow_run(sess, pool_5, 'Forward')
		obj = tf.nn.l2_loss(pool_5)
		grad = tf.gradients(obj, parameters)
		time_tensorflow_run(sess, obj, 'Farward-backward')

run_benchmark()

