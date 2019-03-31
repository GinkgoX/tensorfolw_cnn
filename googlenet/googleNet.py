import tensorflow as tf
form datetime import datetime
import math
import time

#import the predefined googlenet network in slim
slim = tf.contrib.slim
#define the truncated normal distributation
trunc_normal = lambda stddev : tf.truncated_normal_initializer(0.0, stddev)
#define parameters dict
parameters = []

#define inception_v3_arg_scope function to predefined the inception unit parameters 
def inception_v3_arg_scope(weight_decay = 0.00004, stddev = 0.1, batch_norm_var_collection = 'moving_vars'):
	'''
	description:	to predefined the inception unit parameters
	Args:	weight_decay: the L2 regularation weight delay
			stddev: stdandard devience
			batch_norm_var_collection: the BP algorithm training parameters
	Returns:	scope: the 
	'''
	#define the batch_norm_params
	batch_norm_params = {'decay':0.9997, 'epsilon':0.001, 'updates_collections' : tf.GraphKeys.UPDATE_OPS,
						'variables_collections' : {
								'beta' : None, 'gramma' : None, 'moving_mean':[batch_norm_var_collection], 'moving_variance':[batch_norm_var_collection],
							}
						};
	#initialze the slim.conv2d and slim full_connected
	with slim.arg_scope([slim.conv2d, slim.full_connected], weights_regularizer = slim.l2_regularizer(weight_decay)):
		with slim.arg_scope([slim.conv2d], weights_regularizer = tf.truncated_normal_initializer(stddev = stddev),
			activation_fn = tf.nn.relu, normalizer_params = batch_norm_params) as scope:
			return scope
#define inception_v3_base function to setup the (229 * 229) image as tensor
def inception_v3_base(x, scope = None):
	'''
	description: to setup the (229 * 299) image as tensor
	Args:	x : the input data
			scope: the scope output form the last inception
	Returns:	net: the tensor branch sum
				end_points: key points output
	'''
	#define the end_points nodes
	end_points = {}
	#define inceptionV3 network
	with tf.variable_scope(scope, 'InceptionV3', [x]):
		#set slim argments scope 1 padding is VALID
		with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride = 1, padding = 'VALID'):
		#conv 229*229*3 image to 35 * 35 * 192 image with 3 3 * 3 kernal
			net = slim.conv2d(x, 32, [3, 3], stride = 2, scope = 'Conv2d_1a_3x3')
			net = slim.conv2d(net, 32, [3 ,3], scope = 'Conv2d_2a_3X3')
			net = slim.conv2d(net, 64, [3, 3], padding = 'SAME', scope = 'Conv2d_2b_3x3')
			#define max_pool2d
			net = slim.max_pool2d(net, [3, 3], stride = 2, scope = 'MaxPool_3a_3x3')
			net = slim.conv2d(net, 80, [1, 1], scope = 'Conv2d_3b_1x1')
			net = slim.conv2d(net, 192, [3, 3], scope = 'Conv2d_4a_3x3')
			net = slim.max_pool2d(net, [3, 3], stride = 2, scope = 'MaxPool_5a_3x3')
		#set slim argments scope 2 padding is SAME
		with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride = 1, padding = 'SAME'):
			#the first inception model in inception 1
			with tf.variable_scope('Mixed_5b'):
				#channel 1 : 1 * 1 * 64
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 64, [1, 1], scope = 'Conv2d_0a_1x1')
				#channel 2 : 1 * 1 * 1 * 48 -> 1 * 5 * 5 * 64
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 48, [1, 1], scope = 'Conv2d_0a_1x1')
					branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope = 'Conv2d_0b_5x5')
				#channel 3 : 1 * 1 * 1 * 64 -> 2 * 3 * 3 * 96
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.conv2d(net, 64, [1, 1], scope = 'Conv2d_0a_1x1')
					branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope = 'Conv2d_0b_3x3')
					branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope = 'Conv2d_0c_3x3')
				#channel 4 : 1 * 3 * 3 avg_pool2d -> 1 * 1 * 1 * 32
				with tf.variable_scope('Branch_3'):
					branch_3 = slim.avg_pool2d(net, [3, 3], scope = 'AvgPool_0a_3x3')
					branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope = 'Conv2d_0b_1x1')
				#sum of channels: 64 + 64 + 96 + 32 = 256, 1 * 35 * 35 * 256
				net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
			
			#the second inception model in inception 1
			with tf.variable_scope('Mixed_5c'):
				#channel 1 : 1 * 1 * 1 * 64
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 64, [1, 1], scope = 'Conv2d_0a_1x1')
				#channel 2 : 1 * 1 * 1 * 48 -> 1 * 5 * 5 * 64
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 48, [1, 1], scope = 'Conv2d_0b_1x1')
					branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope = 'Conv2d_0b_5x5')
				#channel 3 : 1 * 1 * 1 * 64 ->  2 * 3 * 3 * 96
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.conv2d(net, 64, [1, 1], scope = 'Conv2d_0a_1x1')
					branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope = 'Conv2d_0b_3x3')
					branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope = 'Conv2d_0c_3x3')
				#channel 4 : 1 * 3 * 3 avg_pool2d -> 1 * 1 * 1 * 64
				with tf.variable_scope('Branch_3'):
					branch_3 = slim.avg_pool2d(net, [3, 3], scope = 'AvgPool_0a_3x3')
					branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope = 'Conv2d_0b_1x1')
				#sum of channels: 64 + 64 + 96 + 64 = 288, 1 * 35 * 35 * 288
				net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
			#the third inception in inception 1
			with tf.variable_scope('Mixed_5d'):
				#channel 1 : 1 * 1 * 1 * 64
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 64, [1, 1], scope = 'Conv2d_0a_1x1')
				#channel 2 : 1 * 1 * 1 * 48 -> 1 * 5 * 5 * 64
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 48, [1, 1], scope = 'Conv2d_0a_1x1')
					branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope = 'Conv2d_0b_5x5')
				#channel 3 : 1 * 1 * 1 * 64 -> 2 * 3 * 3 * 96
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.conv2d(net, 64, [1, 1], scope = 'Conv2d_0a_1x1')
					branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope = 'Conv2d_0b_3x3')
					branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope = 'Conv2d_0c_3x3')
				#channel 4 : 1 * 3 * 3 avg_pool2d -> 1 * 1 * 1 * 6
				with tf.variable_scope('Branch_3'):
					branch_3 = slim.avg_pool2d(net, [3, 3], scope = 'AvgPool_0a_3x3')
					branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope = 'Conv2d_0b_1x1')
				#sum of channels: 64 + 64 + 95 + 64 = 288, 1 * 35 * 35 * 288
				net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
			#the first inception in inception 2
			with tf.variable_scope('Mixed_6a'):
				#channel 1 : 1 * 3 * 3 * 384 stride = 2, padding = 'VALID'
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 384, [3, 3], stride = 2, padding = 'VALID', scope = 'Conv2d_1a_1x1')
				#channel 2 : 1 * 1 * 1 * 64 -> 2 * 3 * 3 * 96
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 64, [1, 1], scope = 'Conv2d_1a_1x1')
					branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope = 'Conv2d_1b_3x3')
					branch_1 = slim.conv2d(branch_1, 96, [3, 3], stride = 2, padding = 'VALID', scope = 'Conv2d_1c_1x1')
				#channel 3 : 1 * 3 * 3 avg_pool2d
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.avg_pool2d(net, [3, 3], stride = 2, padding = 'VALID', scope = 'Conv2d_1a_1x1')
				#sum of channels: 1 * 35 * 35 * 256
				net = tf.concat([branch_0, branch_1, branch_2], 3)
			#the second inception in inception 2
			with tf.variable_scope('Mixed_6b'):
				#channel 1 : 1 * 1 * 1 * 192
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 192, [1, 1], scope = 'Conv2d_1a_1x1')
				#channel 2 : 1 * 1 * 1 * 128 -> 1 * 1 * 7 * 128 -> 1 * 7 * 1 * 192
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 128, [1, 1], scope = 'Conv2d_1a_1x1')
					branch_1 = slim.conv2d(branch_1, 128, [1, 7], scope = 'Conv2d_1b_1x7')
					branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope = 'Conv2d_1c_7x1')
				#channel 3 : 1 * 1 * 1 * 128 -> 1 * 1 * 7 * 128 -> 1 * 7 * 1 * 128 -> 1 * 1 * 7 * 192
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_1a_1x1')
					branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_1b_7x1')
					branch_2 = slim.conv2d(branch_2, 128, [1, 7], scope='Conv2d_1c_1x7')
					branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_1d_7x1')
					branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_1e_1x7')
				#channel 4 : 1 * 3 * 3 * avg_pool2d -> 1 * 1 * 1 * 192
				 with tf.variable_scope('Branch_3'):
					branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_1a_3x3')
					branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_1b_1x1')
				#sum of channels: 192 * 4 = 768, 1 * 17 * 17 * 768
				net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
			#the first inception in inception 3
			with tf.variable_scope('Mixed_6c'):
				#channel 1 : 1 * 1 * 1 * 192
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 192, [1, 1], scope = 'Conv2d_1a_1x1')
				#channel 2 : 1 * 1 * 1 * 160 -> 1 * 1 * 7 * 160 -> 1 * 7 * 1 * 192
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 160, [1, 1], scope = 'Conv2d_1a_1x1')
					branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope = 'Conv2d_1b_1x7')
					branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope = 'Conv2d_1c_7x1')
				#channel 3 : 1 * 1 * 1 * 160 -> 1 * 7 * 1 * 160 -> 1 * 7 * 1 * 160 -> 1 * 7 * 1 * 160 -> 1 * 1 * 7 * 192
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_1a_1x1')
					branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_1b_7x1')
					branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_1c_1x7')
					branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_1d_7x1')
					branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_1e_1x7')
				#channel 4 : 1 * 3 * 3 * avg_pool2d -> 1 * 1 * 1 * 192
				 with tf.variable_scope('Branch_3'):
					branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_1a_3x3')
					branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_1b_1x1')
				#sum of channels: 192 * 4 = 768, 1 * 17 * 17 * 768
				net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
			#the first inception in inception 4
			with tf.variable_scope('Mixed_6d'):
				#channel 1 : 1 * 1 * 1 * 192
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 192, [1, 1], scope = 'Conv2d_1a_1x1')
				#channel 2 : 1 * 1 * 1 * 160 -> 1 * 1 * 7 * 160 -> 1 * 7 * 1 * 192
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 160, [1, 1], scope = 'Conv2d_1a_1x1')
					branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope = 'Conv2d_1b_1x7')
					branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope = 'Conv2d_1c_7x1')
				#channel 3 : 1 * 1 * 1 * 160 -> 1 * 7 * 1 * 160 -> 1 * 7 * 1 * 160 -> 1 * 7 * 1 * 160 -> 1 * 1 * 7 * 192
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_1a_1x1')
					branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_1b_7x1')
					branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_1c_1x7')
					branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_1d_7x1')
					branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_1e_1x7')
				#channel 4 : 1 * 3 * 3 * avg_pool2d -> 1 * 1 * 1 * 192
				 with tf.variable_scope('Branch_3'):
					branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_1a_3x3')
					branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_1b_1x1')
				#sum of channels: 192 * 4 = 768, 1 * 17 * 17 * 768
				net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
			#the first inception in inception 5
			with tf.variable_scope('Mixed_6e'):
				#channel 1 : 1 * 1 * 1 * 192
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 192, [1, 1], scope = 'Conv2d_1a_1x1')
				#channel 2 : 1 * 1 * 1 * 192 -> 1 * 1 * 7 * 192 -> 1 * 7 * 1 * 192
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 192, [1, 1], scope = 'Conv2d_1a_1x1')
					branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope = 'Conv2d_1b_1x7')
					branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope = 'Conv2d_1c_7x1')
				#channel 3 : 1 * 1 * 1 * 192 -> 1 * 7 * 1 * 192 -> 1 * 7 * 1 * 192 -> 1 * 7 * 1 * 192 -> 1 * 1 * 7 * 192
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_1a_1x1')
					branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_1b_7x1')
					branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_1c_1x7')
					branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_1d_7x1')
					branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_1e_1x7')
				#channel 4 : 1 * 3 * 3 * avg_pool2d -> 1 * 1 * 1 * 192
				 with tf.variable_scope('Branch_3'):
					branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_1a_3x3')
					branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_1b_1x1')
				#sum of channels: 192 * 4 = 768, 1 * 17 * 17 * 768
				net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
			#save the Mixed_6e into end_points
			end_points['Mixed_6e'] = net
			# Inception 3
			with tf.variable_scope('Mixed_7c'):
				# channel 1 : 1 * 320 * 1 * 1
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
				#channel 2 : 1 * 1 * 1 * 384(1 * 3 -> 3 * 1) 
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
					branch_1 = tf.concat([slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'),slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0c_3x1')], 3)
				#channel 23 : 1 * 1 * 1 * 448, 1 * 3 * 3 * 384 (1 * 3 -> 3 * 1)
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
					branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
					branch_2 = tf.concat([slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'),slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')], 3)
				#channel 4: 1 * 3 * 3 avg_pool2d, 1 * 1 * 1 * 192
				with tf.variable_scope('Branch_3'):
					branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
					branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
				#sum of channels: 320 + 768 + 768 + 192 = 2048
				net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
			return net, end_points

# total region avg_pool2d
def inception_v3(inputs,num_classes=1000,is_training=True,dropout_keep_prob=0.8,prediction_fn=slim.softmax,spatial_squeeze=True,reuse=None,scope='InceptionV3'): 
	with tf.variable_scope(scope, 'InceptionV3', [inputs, num_classes], reuse=reuse) as scope:
		with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
			net, end_points = inception_v3_base(inputs, scope=scope)
			#set the slim argments scope parameters
			with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
				aux_logits = end_points['Mixed_6e']
				#AuxLogits nodes
				with tf.variable_scope('AuxLogits'):
					# 1 * 5 * 5 avg_pool2d -> 1 * 1 * 1 * 128 -> 1 * 5 * 5 * 768 -> 1 * 1 * 1 * class(1000)
					aux_logits = slim.avg_pool2d(aux_logits, [5, 5], stride=3, padding='VALID', scope='AvgPool_1a_5x5')
					aux_logits = slim.conv2d(aux_logits, 128, [1, 1], scope='Conv2d_1b_1x1')
					aux_logits = slim.conv2d(aux_logits,768, [5, 5], weights_initializer=trunc_normal(0.01), padding='VALID', scope='Conv2d_2a_5x5')
					aux_logits = slim.conv2d(aux_logits, num_classes, [1, 1], activation_fn=None,
												normalizer_fn=None, weights_initializer=trunc_normal(0.001), scope='Conv2d_2b_1x1')
					if spatial_squeeze:
						#squeeze operation
						aux_logits = tf.squeeze(aux_logits, [1, 2], name='SpatialSqueeze')
					end_points['AuxLogits'] = aux_logits
			#define the class result scope
			with tf.variable_scope('Logits'):
				# 1 * 8 * 8 avg_pool2d
				net = slim.avg_pool2d(net, [8, 8], padding='VALID', scope='AvgPool_1a_8x8')
				# dropout layer
				net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
				end_points['PreLogits'] = net
				logits = slim.conv2d(net, num_classes, [1, 1], activation_fn= None, normalizer_fn=None,scope='Conv2d_1c_1x1')
				if spatial_squeeze:
					#squeeze operation
					logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
			end_points['Logits'] = logits
			#softmax prediction
			end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
	return logits, end_points



import math
from datetime import datetime
import time

#evaluate the time consuming
def time_tensorflow_run(session, target, info_string):
	num_steps_burn_in = 10
	total_duration = 0.0
	total_duration_squared = 0.0

	for i in range(num_batches + num_steps_burn_in):
		tart_time = time.time()
		_ = session.run(target)
		duration = time.time()- start_time
		if i >= num_steps_burn_in:
			if not i % 10:
				print ('%s: step %d, duration = %.3f' % (datetime.now().strftime('%X'), i - num_steps_burn_in, duration))
			total_duration += duration
			total_duration_squared += duration * duration
	# calculate the aveage time consuming and standand variance
	mn = total_duration / num_batches
	vr = total_duration_squared / num_batches - mn * mn
	sd = math.sqrt(vr)
	print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' % (datetime.now().strftime('%X'), info_string, num_batches, mn, sd))


#main process
if __name__ == '__main__':
	batch_size = 32
	height, width = 299, 299
	inputs = tf.random_uniform((batch_size, height, width, 3))
	with slim.arg_scope(inception_v3_arg_scope()):
		#get logits and end_points result
		logits, end_points = inception_v3(inputs, is_training=False)
	#create the Session
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)
	num_batches = 100
	time_tensorflow_run(sess, logits, 'Forward')



