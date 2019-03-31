import tensorflow as tf
import numpy as np
import setting
import models
import scipy.misc

#define loss function to calculate the sum loss of content loss and style loss
def loss(sess, model):
	'''
	description: to calculate the sum loss of content loss and style loss
	Args:	sess: tf Session
			model: the vgg_19 model parameters
	Returns:	the loss sum
	'''
	#access content layer
	content_layers = setting.CONTENT_LOSS_LAYERS
	#define the input image as content
	sess.run(tf.assign(model.net['input'], model.content))
	#calculate the content loss
	content_loss = 0.0
	#access the weights and biases in layers defined in vgg_19
	for layer_name, weights in content_layers:
		#extract the feature matrx in layer_name for content image
		p = sess.run(model.net[layer_name])
		#extract the feature matrx in layer_name for noise image
		x = model.net[layer_name]
		# M = length * width
		M = p.shape[1] * p.shape[2]
		# N = channel numbers
		N = p.shape[3]
		#calculate the content loss
		content_loss += (1.0 / (2 * M * N)) * tf.reduce_sum(tf.pow(p - x, 2))*weights
	content_loss /= len(content_layers)
	
	#access style layer
	style_layers = setting.STYLE_LOSS_LAYERS
	#define the input image as style
	sess.run(tf.assign(model.net['input'], model.style))
	#calculate the style loss
	style_loss = 0.0
	for layer_name, weights in style_layers:
		#extract the feature matrx in layer_name for style image
		a = sess.run(model.net[layer_name])
		#extract the feature max in layer_name for noise image
		x = model.net[layer_name]
		# M = length * width
		M = a.shape[1] * a.shape[2]
		# N = channel numbers
		N = a.shape[3]
		# A = gram(a) [style image gram feature matrx]
		A = gram(a, M, N)
		# G = gram(x) [noise image gram feature matrx]
		G = gram(x, M, N)
		#calculate the style_loss
		style_loss += (1.0 / (4 * M * M * N * N)) * tf.reduce_sum(tf.pow(G - A, 2)) * weights
	style_loss /= len(style_layers)
	#the total loss result
	loss = setting.ALPHA * content_loss + setting.BETA * style_loss
	return loss


#define gram function to calculate the g = transpose(x)*x
def gram(x, size, deepth):
	'''
	description: to calculate the gram result
	Args:	x: the input data
			size: the result (length * width)
			deepth: the channel numbers
	Returns:	g = transpose(x) * x
	'''
	x = tf.reshape(x, (size, deepth))
	g = tf.matmul(tf.transpose(x), x)
	return g

#define train function to train the model
def train():
	'''
	description: to train the model
	Args:	None
	Returns:	None
	'''
	model = models.model(setting.CONTENT_IMAGE, setting.STYLE_IMAGE)
	with tf.Session() as sess:
		#intialize the global variables
		sess.run(tf.global_variables_initializer())
		#define cost
		cost = loss(sess, model)
		#define optimizer
		optimizer = tf.train.AdamOptimizer(1.0).minimize(cost)
		#intialize the global variables sine the new operation
		sess.run(tf.global_variables_initializer())
		#train the model with noise image
		sess.run(tf.assign(model.net['input'], model.random_img))
		#define the train setps
		for step in range(setting.TRAIN_STEPS):
			#define BP once
			sess.run(optimizer)
			#output the trainning results
			if step % 50 == 0:
				print('step{} is done .'.format(step))
				#access the generated image
				img = sess.run(model.net['input'])
				#recover the image with adding the image mean value
				img += img + setting.IMAGE_MEAN_VALUE
				#get the batch 0 demension
				img = img[0]
				#recover the float32 image to int image in [0, 255]
				img = np.clip(img, 0, 255).astype(np.uint8)
				#save the image
				scipy.misc.imsave('{}-{}.jpg'.format(setting.OUTPUT_IMAGE, step), img)

		#save the finnal result
		img = sess.run(model.net['input'])
		#recover the image with adding the image mean value
		img += img + setting.IMAGE_MEAN_VALUE
		#get the batch 0 demension
		img = img[0]
		#recover the float32 image to int image in [0, 255]
		img = np.clip(img, 0, 255).astype(np.uint8)
		#save the image
		scipy.misc.imsave('{}.jpg'.format(setting.OUTPUT_IMAGE), img)

if __name__ == '__main__':
	train()
