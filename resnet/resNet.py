import tensorflow as tf
import collections
from datetime import datetime
import math
import time

slim = tf.contrib.slim

#define Block class to build different conv layer Block
class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
	'''
	description : to build different conv layer Block
	'''

#define subsample function to excute max_pool2d operation
def subsample(x, factor, scope):
	'''
	description : to excute max_pool2d operation
	Args : 	x : the input data
			factor : the sampling factor
			scope : the name scope
	Returns: x or max_pool2d result depending on the factor
	'''
	if factor == 1:
		return x
	else:
		return slim.max_pool2d(x, [1, 1], stride = factor, scope = scope)

#define conv2d_same to excute conv operation with different padding method
def conv2d_same(x, channel, kernal, stride, scope = None):
	'''
	description:	
	Args	x : the input data
			channel : the output channel
			kernal : the kernal(filter) size
			stride : the conv stride
			scope : the name scope
	Returns : conv2d result
	'''
	if stride == 1 :
		return slim.conv2d(x, channel, kernal, stride, padding = 'SAME', scope = scope)
	else:
		pad_total = kernal - 1
		pad_beg = pad_total // 2
		pad_end = pad_total - pad_beg
		x = tf.pad(x, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
		return slim.conv2d(x, channel, kernal, stride = stride, padding = 'VALID', scope = scope)

@slim.add_arg_scope
#define stack_block_dense function to stack the block
def stack_block_dense(x, blocks, out_collections = None):
	'''
	description: to stack the block
	Args:	x : the input data
			blocks: the name scope / block form Block
			collections: the end_points from collection
	Returns:	the densed output tensor
	'''
	for block in blocks:
		with tf.variable_scope(block.scope, 'block', [x]) as scope:
			for i, unit in enumerate(block.args):
				with tf.variable_scope('unit_%d'%(i + 1), values = [x]):
					depth, depth_bottlenect, stride = unit
					#residual learning unit
					x = block.unit_fn(x, depth = depth, depth_bottlenect = depth_bottlenect, stride = stride)
	return slim.utils.collect_named_outputs(out_collections, scope.name, x)

#define resnet_arg_scope function to predefine the parameters in resnet
def resnet_arg_scope(weight_decay = 0.0001, batch_norm_decay = 0.997, batch_norm_epsilon = 1e-5, batch_norm_scale = True, is_training = True):
	'''
	description: to predefine the parameters in resnet
	Args:	weight_decay: weight decay rate
			batch_norm_decay: BN decay rate 
			batch_norm_epsilon: BN epsilon parameter
			batch_norm_scale: BN scale parameter
			is_training:
	'''
	#define the BN parameters
	batch_norm_params = {
		'decay' : batch_norm_decay,
		'epsilon' : batch_norm_epsilon,
		'scale' : batch_norm_scale,
		'is_training' : is_training,
		'update_collections' : tf.GraphKeys.UPDATE_OPS,
	}

	#set the slim args scope
	with slim.arg_scope(
			[slim.conv2d],
			weights_regularizer = slim.l2_regularizer(weight_decay),
			weights_initializer = slim.variance_scaling_initializer(),
			activation_fn = tf.nn.relu,
			normalizer_fn = slim.batch_norm,
			normalizer_params = batch_norm_params):
		with slim.arg_scope([slim.batch_norm], **batch_norm_params):
			with slim.arg_scope([slim.max_pool2d], padding = 'SAME') as arg_scope:
				return arg_scope

@slim.add_arg_scope
#define bottlenect function to initialize the bottlenect residual learning unit
def bottlenect(x, depth, depth_bottlenect, stride, out_collections = None, scope = None):
	'''
	description: to initialize the bottlenect residual learning unit
	Args:	x : the input data
			depth : the output channel
			depth_bottlenect : bottlenect residual learning unit
			stride : the stride for conv operation
			collections : the end_points from collection
			scope : the name scope
	Returns:	H(x) = f(x) + x
	'''
	with tf.variable_scope(scope, 'bottlenect_v2', [x]) as scope:
		# access the input channel
		depth_in = slim.units.last_dimension(x.get_shape(), min_rank = 4)
		# excute the relu and BN operation for preactivation [the V2 method]
		preactivation = slim.batch_norm(x, activation_fn = tf.nn.relu, scope = 'preactivation')
		#define shortcut operation
		if depth == depth_in:
			shortcut = subsample(x, stride, 'shortcut')
		else:
			shortcut = slim.conv2d(preactivation, depth, [1, 1], stride = stride, normalizer_fn = None, activation_fn = None, scope = 'shortcut')
		#calculate f_x
		F_x = slim.conv2d(preactivation, depth_bottlenect, [1 ,1], stride = 1, scope = 'conv1')
		F_x = conv2d_same(residual, depth_bottlenect, 3, stride, scope = 'conv2')
		F_X = slim.conv2d(residual, depth, [1, 1], stride = 1, normalizer_fn = None, activation_fn = None, scope = 'conv3')
		#calculate the output
		H_x = F_x + shortcut
		return slim.utils.collect_named_outputs(out_collections, scope.name, H_x)

#define resnet_v2 main function to load the blocks
def resnet_v2(x, blocks, num_class = None, global_pool = None, include_root_block = True, reuse = None, scope = None):
	'''
	description : to load blocks
	Args:	x : the input data
			blocks : the Block object
			num_class : the number of class
			global_pool : global avg pool better than avg_pool directly
			include_root_block : whether include root block
			reuse : whether reuse the block
			scope : the name scope
	Returns:	net : the conv network result
				end_points : the output collection result
	'''
	with tf.variable_scope(scope, 'resnet_v2', [x], reuse = reuse) as scope:
		end_points = scope.original_name_scope + '_end_points'
		with slim.arg_scope([slim.conv2d, bottlenect, stack_block_dense], out_collections = end_points):
			net = x
			if include_root_block:
				with slim.arg_scope([slim.conv2d], activation_fn = None, normalizer_fn = None):
					net = conv2d_same(net, 64, 7, stride = 2, scope = 'conv1')
				net = slim.max_pool2d(net, [3, 3], stride = 2, scope = 'conv1')
			net = stack_block_dense(net, blocks)
			net = slim.batch_norm(net, activation_fn = tf.nn.relu, scope = 'postnorm')
			if global_pool:
				net = tf.reduce_mean(net, [1, 3], scope = 'pool5', keep_dims = True)
			if num_class is not None:
				net = slim.conv2d(net, num_class, [1, 1], activation_fn = None, normalizer_fn = None, scope = 'logits')
			end_points = slim.utils.convert_collection_to_dict(end_points)
			if num_class is not None:
				end_points['prediction'] = slim.sofmax(net, scope = 'prediction')
			return net, end_points

#define resnet_v2_50 network architecture
def resnet_v2_50(x, num_class = None, global_pool = True, reuse = True, scope = 'resnet_v2_50'):
	'''
	description : to build the resnet_v2_50 model
	Args :	x : the input data
			num_class : the number of class
			global_pool : global average pool
			reuse : whether reuse block
			scope : the name scope
	Returns :	net : the network model
				end_points : the end_points collection
	'''
	blocks = [
				Block('block1', bottlenect, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
				Block('block2', bottlenect, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
				Block('block3', bottlenect, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
				Block('block4', bottlenect, [(2048, 512, 1)] * 3)]
	return resnet_v2(x, blocks, num_class, global_pool, include_root_block = True, reuse = reuse, scope = scope)

#define resnet_v2_101 network architecture
def resnet_v2_101(x, num_class = None, global_pool = True, reuse = True, scope = 'resnet_v2_101'):
	'''
	description : to build the resnet_v2_101 model
	Args :	x : the input data
			num_class : the number of class
			global_pool : global average pool
			reuse : whether reuse block
			scope : the name scope
	Returns :	net : the network model
				end_points : the end_points collection
	'''
	blocks = [
				Block('block1', bottlenect, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
				Block('block2', bottlenect, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
				Block('block3', bottlenect, [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
				Block('block4', bottlenect, [(2048, 512, 1)] * 3)]
	return resnet_v2(x, blocks, num_class, global_pool, include_root_block = True, reuse = reuse, scope = scope)

#define resnet_v2_152 network architecture
def resnet_v2_152(x, num_class = None, global_pool = True, reuse = True, scope = 'resnet_v2_152'):
	'''
	description : to build the resnet_v2_152 model
	Args :	x : the input data
			num_class : the number of class
			global_pool : global average pool
			reuse : whether reuse block
			scope : the name scope
	Returns :	net : the network model
				end_points : the end_points collection
	'''
	blocks = [
				Block('block1', bottlenect, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
				Block('block2', bottlenect, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
				Block('block3', bottlenect, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
				Block('block4', bottlenect, [(2048, 512, 1)] * 3)]
	return resnet_v2(x, blocks, num_class, global_pool, include_root_block = True, reuse = reuse, scope = scope)

#define resnet_v2_200 network architecture
def resnet_v2_200(x, num_class = None, global_pool = True, reuse = True, scope = 'resnet_v2_200'):
	'''
	description : to build the resnet_v2_152 model
	Args :	x : the input data
			num_class : the number of class
			global_pool : global average pool
			reuse : whether reuse block
			scope : the name scope
	Returns :	net : the network model
				end_points : the end_points collection
	'''
	blocks = [
				Block('block1', bottlenect, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
				Block('block2', bottlenect, [(512, 128, 1)] * 23 + [(512, 128, 2)]),
				Block('block3', bottlenect, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
				Block('block4', bottlenect, [(2048, 512, 1)] * 3)]
	return resnet_v2(x, blocks, num_class, global_pool, include_root_block = True, reuse = reuse, scope = scope)

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
			total_duration_squared += duration * duration
	#calculate avarage time consuming
	ave_time = total_duration / num_batches
	#calculate strandand deviance
	var = ave_time*ave_time - total_duration_squared / num_batches
	std_dev = math.sqrt(var)
	print('%s,	%s across %d steps, %.3f, +/- %.3f sec/batch'%(datetime.now(), test_name, num_batches, ave_time, std_dev))

#define batch size
batch_size = 32
#define image_size
image_size = 224
#generate the random image
x = tf.random_uniform((batch_size, image_size, image_size, 3))
#access the net result
with slim.arg_scope(resnet_arg_scope(is_training = False)):
	net, end_points = resnet_v2_152(x, 1000)

#initialize the global variable
init = tf.global_variables_initializer()
#create Session
sess = tf.Session()
sess.run(init)
num_batches = 100
#estimate the time consuming
time_tensorflow_run(sess, net, 'Forward')


