import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

#define identity_block to realize the residual unit
def identity_block(X, k_stride, k_size, stage, block):
	'''
	description : to realize the residual unit
	Args : 	X -- the input data
			k_stride -- the kernel stride
			k_size -- the kernel size
			stage -- the network position
			block -- the layer name
	Returns : the activation result of X
	'''
	#define the name bias
	conv_name_base = 'res' + str(stage) + block + 'branch'
	bn_name_base = 'bn' + str(stage) + block + 'branch'
	#retrive the filters
	F1, F2, F3 = k_size
	#copy the input data for final adding usage
	X_shortcut = X
	
	#1 component for main path conv -> bn -> relu
	X = Conv2D(filters = F1,  kernel_size = (1, 1), strides = (1, 1), padding = 'valid', 
				name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed = 0))(X)
	X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
	X = Activation('relu')(X)
	
	#2 component for main path conv ->bn -> relu
	X = Conv2D(filters = F2,  kernel_size = (k_stride, k_stride), strides = (1, 1), padding = 'same', 
				name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed = 0))(X)
	X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
	X = Activation('relu')(X)
	
	#3 component for main path conv -> bn
	X = Conv2D(filters = F3,  kernel_size = (1, 1), strides = (1, 1), padding = 'valid', 
				name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed = 0))(X)
	X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
	
	#shortcut path
	X = Add()([X, X_shortcut])
	X = Activation('relu')(X)
	return X

#define convolutional_block to excute the convolutional operation
def convolutional_block(X, k_stride, k_size, stage, block, stride = 2):
	'''
	description : to excute the convolutional operation
	Args :	X -- the input data
			k_stride -- the kernel stride
			k_size -- the kernel size
			stage -- the stage name of layer
			block -- the block name
			stride -- different stride from k_stride
	Returns:	X -- the convolutional result of X
	'''
	#define name bias
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'
	#retrive filters
	F1, F2, F3 = k_size
	#copy the input X
	X_shortcut = X
	#1 component of main path
	X = Conv2D(F1, (1, 1), strides = (stride, stride), 
					name = conv_name_base + '2a', padding = 'valid', kernel_initializer = glorot_uniform(seed = 0))(X)
	X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
	X = Activation('relu')(X)
	
	#2 component of main path
	X = Conv2D(F2, (k_stride, k_stride), strides = (1, 1), 
					name = conv_name_base + '2b', padding = 'same', kernel_initializer = glorot_uniform(seed = 0))(X)
	X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
	X = Activation('relu')(X)
	
	#3 component of main path
	X = Conv2D(F3, (1, 1), strides = (1, 1), 
					name = conv_name_base +'2c', padding = 'valid', kernel_initializer = glorot_uniform(seed = 0))(X)
	X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
	
	#shortcut
	X_shortcut = Conv2D(F3, (1, 1), strides = (stride, stride), 
							name = conv_name_base + '1',padding = 'valid', kernel_initializer = glorot_uniform(seed = 0))(X_shortcut)
	X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)
	
	#final main path
	X = Add()([X, X_shortcut])
	X = Activation('relu')(X)
	return X

#define resNet50 function to set up the resNet50 network
def resNet50(input_shape = (64, 64 ,3), classes = 6):
	'''
	description : to set up the resNet50 network
	Args :	input_shape  -- the input data
			classes -- the number of classes
	Returns : model -- the keras model
	'''
	#define the input as a tensor with shape input_shape
	X_input = Input(input_shape)
	#zero padding
	X = ZeroPadding2D((3, 3))(X_input)
	#stage 1
	X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed = 0))(X)
	X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
	X = Activation('relu')(X)
	X = MaxPooling2D((3, 3), strides = (2, 2))(X)
	#stage 2
	X = convolutional_block(X, k_stride = 3, k_size = [64, 64, 256], stage = 2, block = 'a', stride = 1)
	X = identity_block(X, 3, [64, 64, 256], stage = 2, block = 'b')
	X = identity_block(X, 3, [64, 64, 256], stage = 2, block = 'c')
	#stage 3
	X = convolutional_block(X, k_stride = 3, k_size = [128, 128, 512], stage = 3, block = 'a', stride = 2)
	X = identity_block(X, 3, [128, 128, 512], stage = 3, block = 'b')
	X = identity_block(X, 3, [128, 128, 512], stage = 3, block = 'c')
	X = identity_block(X, 3, [128, 128, 512], stage = 3, block = 'd')
	#stage 4
	X = convolutional_block(X, k_stride = 3, k_size = [256, 256, 1024], stage = 4, block = 'a', stride = 2)
	X = identity_block(X, 3, [256, 256, 1024], stage = 4, block = 'b')
	X = identity_block(X, 3, [256, 256, 1024], stage = 4, block = 'c')
	X = identity_block(X, 3, [256, 256, 1024], stage = 4, block = 'd')
	X = identity_block(X, 3, [256, 256, 1024], stage = 4, block = 'e')
	X = identity_block(X, 3, [256, 256, 1024], stage = 4, block = 'f')
	#stage 5
	X = convolutional_block(X, k_stride = 3, k_size = [512, 512, 2048], stage = 5, block = 'a', stride = 2)
	X = identity_block(X, 3, [512, 512, 2048], stage = 5, block = 'b')
	X = identity_block(X, 3, [512, 512, 2048], stage = 5, block = 'c')
	#average pooling
	X = AveragePooling2D((2, 2), name = 'avg_pool')(X)
	#output label
	X = Flatten()(X)
	X = Dense(classes, activation = 'softmax', name = 'full_connection' + str(classes), kernel_initializer = glorot_uniform(seed = 0))(X)
	#create model
	model = Model(inputs = X_input, outputs = X, name = 'resNet50')
	return model

#initialize the resNet50 model
model = resNet50(input_shape = (64, 64 ,3), classes = 6)
#compile model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#save the model
model.save('resNet50.h5')
#get the data
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
#normalize the image
X_train = X_train_orig / 255
X_test = X_test_orig / 255
#conver Y to one hot codding
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
#feed date
model.fit(X_train, Y_train, epochs = 10, batch_size = 32)
#evaluate the model
preds = model.evaluate(X_test, Y_test)
print('Loss : ' + str(preds[0]))
print('accuracy : ' + str(preds[1]))
#print the network
model.summary()
#plot the model
plot_model(model, to_file = 'resNet50.jpg')

#define image path
img_path = 'test.jpg'
#load image
img = image.load_img(img_path, target_size = (64, 64))
#convert the image demension
x = image.img_to_array(img)
#expend the image dims
x = np.expand_dims(x, axis = 0)
#preprocess the input
x = preprocess_input(x)
#print the preprocess input shape
print('input image shape : ', x.shape)
#read the image
test = scipy.misc.imread(img_path)
#imshow
imshow(test)
#predict the result
print(model.predict(x))

