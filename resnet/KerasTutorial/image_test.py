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
from keras.models import load_model

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

#load model
model = load_model('resNet50.h5')
#define image path
img_path = 'test1.jpg'
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

