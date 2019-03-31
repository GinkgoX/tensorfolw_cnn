import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

#get the data info
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
#normalize the X_train_orig
X_train = X_train_orig / 255
#normalize the X_test_orig
X_test = X_test_orig / 255
#Reshape Y
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

#define happyModel to set up a neurial network
def happyModel(input_shape):
	'''
	description : to set up a neurial network
	Args : 	input_shape  -- the input shape
	Returns : 	model	-- the keras model instance
	'''
	#define the input placeholder as a tensor with shape input_shape
	X_input = Input(input_shape)
	#padding the X_input boder with zero
	X = ZeroPadding2D((3, 3))(X_input)
	#adapt conv -> bn -> relu to access the X feature
	X = Conv2D(32, (7, 7), strides = (1 ,1), name = 'conv0')(X)
	
	X = BatchNormalization(axis = 3, name = 'bn0')(X)
	X = Activation('relu')(X)
	#maxpooling
	X = MaxPooling2D((2, 2), name = 'max_pool')(X)
	#flatten and dense operation
	X = Flatten()(X)
	X = Dense(1, activation = 'sigmoid', name = 'fc')(X)
	#output the model
	model = Model(inputs = X_input, outputs = X, name = 'happyModel')
	return model

#call the model
happyModel = happyModel(X_train.shape[1:])
#compile and optimize the model
happyModel.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#train the model
happyModel.fit(x = X_train, y = Y_train, epochs = 10, batch_size = 32)
#evaluate the model
preds = happyModel.evaluate(X_test, Y_test)
#print Loss
print('Loss : ' + str(preds[0]))
#print accuracy
print('accuracy : ' + str(preds[1]))
#print the network
happyModel.summary()
#plot the model
plot_model(happyModel, to_file = 'happyModel.jpg')
