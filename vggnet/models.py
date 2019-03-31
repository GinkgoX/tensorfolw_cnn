import tensorflow as tf
import numpy as np
import setting
import scipy.io
import scipy.misc

#define model class to build vgg_19 network
class model(object):
	'''
	description: to build vgg_19 network
	Funcs:	__init__(self, content_path, style_path): initialize parameters
			vggnet(self): construct vgg_19 network
			conv_relu(self, x, wb): relu activation
			pool(self, x): max_pool operation
			get_wb(self, layers, i): get the vgg parameter in i-th layer
			get_random_img(self): get noise image
			load_img(self, path): load the image form path
	'''
	def __init__(self, content_path, style_path):
		'''
		description: to initialize parameters
		Args:	content_path: content image path
				style_path: style image path
		Returns: None
		'''
		#get content image path
		self.content = self.load_img(content_path)
		#get style image path
		self.style = self.load_img(style_path)
		#get random noise image
		self.random_img = self.get_random_img()
		#set up the vgg network
		self.net = self.vggnet()

	def vggnet(self):
		'''
		description: to construct vgg_19 network
		Args:	self
		Returns :	net: the vgg_19 pretrained network without full connection layers
		'''
		#get the prtrained vgg-19.mat data
		vgg = scipy.io.loadmat(setting.VGG_MODEL_PATH)
		vgg_layers = vgg['layers'][0]
		#initialize net dict
		net = {}
		#use vgg19 pretrained model parameters to train input image without full connection layers
		net['input'] = tf.Variable(np.zeros([1, setting.IMAGE_HEIGHT, setting.IMAGE_WIDTH, 3]), dtype = tf.float32)
		net['conv1_1'] = self.conv_relu(net['input'], self.get_wb(vgg_layers, 0))
		net['conv1_2'] = self.conv_relu(net['conv1_1'], self.get_wb(vgg_layers, 2))
		net['pool1'] = self.pool(net['conv1_2'])
		
		net['conv2_1'] = self.conv_relu(net['pool1'], self.get_wb(vgg_layers, 5))
		net['conv2_2'] = self.conv_relu(net['conv2_1'], self.get_wb(vgg_layers, 7))
		net['pool2'] = self.pool(net['conv2_2'])
		
		net['conv3_1'] = self.conv_relu(net['pool2'], self.get_wb(vgg_layers, 10))
		net['conv3_2'] = self.conv_relu(net['conv3_1'], self.get_wb(vgg_layers, 12))
		net['conv3_3'] = self.conv_relu(net['conv3_2'], self.get_wb(vgg_layers, 14))
		net['conv3_4'] = self.conv_relu(net['conv3_3'], self.get_wb(vgg_layers, 16))
		net['pool3'] = self.pool(net['conv3_4'])
		
		net['conv4_1'] = self.conv_relu(net['pool3'], self.get_wb(vgg_layers, 19))
		net['conv4_2'] = self.conv_relu(net['conv4_1'], self.get_wb(vgg_layers, 21))
		net['conv4_3'] = self.conv_relu(net['conv4_2'], self.get_wb(vgg_layers, 23))
		net['conv4_4'] = self.conv_relu(net['conv4_3'], self.get_wb(vgg_layers, 25))
		net['pool4'] = self.pool(net['conv4_4'])
		
		net['conv5_1'] = self.conv_relu(net['pool4'], self.get_wb(vgg_layers,28))
		net['conv5_2'] = self.conv_relu(net['conv5_1'], self.get_wb(vgg_layers, 30))
		net['conv5_3'] = self.conv_relu(net['conv5_2'], self.get_wb(vgg_layers, 32))
		net['conv5_4'] = self.conv_relu(net['conv5_3'], self.get_wb(vgg_layers, 34))
		net['pool5'] = self.pool(net['conv5_4'])
		return net

	def conv_relu(self, x, wb):
		'''
		description: to excute relu activation
		Args:	x: the input data
				wb: weights and biases array
		Returns:	result of relu(x*wb[0] + wb[1])
		'''
		#excute conv2d operation
		conv = tf.nn.conv2d(x, wb[0], strides = [1, 1, 1, 1], padding = 'SAME')
		#relu activation
		relu = tf.nn.relu(conv + wb[1])
		return relu

	def pool(self, x):
		'''
		description: to excute max_pool operation
		Args:	x: the input data
		Returns:	the results of max_pool
		'''
		return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

	def get_wb(self, layers, i):
		'''
		description: to get weights and biases array
		Args:	layers: the pretrained vgg_19 layers
				i: the i-th layer
		Returns:	the w and b array
		'''
		
		w = tf.constant(layers[i][0][0][0][0][0])
		bias = layers[i][0][0][0][0][1]
		b = tf.constant(np.reshape(bias, (bias.size)))
		return w, b

	def get_random_img(self):
		'''
		description: to generate the random noise image
		Args:	self:
		Returns:	random_img
		'''
		noise_img = np.random.uniform(-20, 20, [1, setting.IMAGE_HEIGHT, setting.IMAGE_WIDTH, 3])
		random_img = noise_img * setting.NOIZE + self.content * (1 - setting.NOIZE)
		return random_img

	def load_img(self, path):
		'''
		description: to load the image form path
		Args:	self
				path: the file path
		Returns:	image
		'''
		img = scipy.misc.imread(path)
		img = scipy.misc.imresize(img, [setting.IMAGE_HEIGHT, setting.IMAGE_WIDTH])
		img = np.reshape(img, [1, setting.IMAGE_HEIGHT, setting.IMAGE_WIDTH, 3])
		return img

if __name__ == '__main__':
	model(setting.CONTENT_IMAGE, setting.STYLE_IMAGE)
