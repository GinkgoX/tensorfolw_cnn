import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
import h5py
import scipy

def get_files(file_dir):
	
	train = []
	label = []
	
	for file in os.listdir(file_dir+'/daisy'):
			train.append(file_dir +'/daisy'+'/'+ file)
			label.append(0)     #添加标签，该类标签为0，此为2分类例子，多类别识别问题自行添加
	for file in os.listdir(file_dir+'/dandelion'):
			train.append(file_dir +'/dandelion'+'/'+file)
			label.append(1)
	for file in os.listdir(file_dir+'/dandelion'):
			train.append(file_dir +'/dandelion'+'/'+file)
			label.append(2)
	for file in os.listdir(file_dir+'/roses'):
			train.append(file_dir +'/roses'+'/'+file)
			label.append(3)
	for file in os.listdir(file_dir+'/sunflowers'):
			train.append(file_dir +'/sunflowers'+'/'+file)
			label.append(4)
	for file in os.listdir(file_dir+'/tulips'):
			train.append(file_dir +'/tulips'+'/'+file)
			label.append(5)
	
	#利用shuffle打乱顺序
	temp = np.array([train, label])
	temp = temp.transpose()
	np.random.shuffle(temp)
 
	#从打乱的temp中再取出list（img和lab）
	image_list = list(temp[:, 0])
	label_list = list(temp[:, 1])
	label_list = [int(i) for i in label_list]
	
	return image_list,label_list
	 #返回两个list 分别为图片文件名及其标签  顺序已被打乱

train_dir = './flower_photos'
image_list,label_list = get_files(train_dir)
 
print(len(image_list))
print(len(label_list))

#450为数据长度的20%
Train_image =  np.random.rand(len(image_list)-450, 64, 64, 3).astype('float32')
Train_label = np.random.rand(len(image_list)-450, 1).astype('float32')
 
Test_image =  np.random.rand(450, 64, 64, 3).astype('float32')
Test_label = np.random.rand(450, 1).astype('float32')

for i in range(len(image_list)-450):
    Train_image[i] = np.array(plt.imread(image_list[i]))
    Train_label[i] = np.array(label_list[i])
 
for i in range(len(image_list)-450, len(image_list)):
    Test_image[i+450-len(image_list)] = np.array(plt.imread(image_list[i]))
    Test_label[i+450-len(image_list)] = np.array(label_list[i])
 
# Create a new file
f = h5py.File('data.h5', 'w')
f.create_dataset('X_train', data=Train_image)
f.create_dataset('y_train', data=Train_label)
f.create_dataset('X_test', data=Test_image)
f.create_dataset('y_test', data=Test_label)
f.close()
'''
# Load hdf5 dataset
train_dataset = h5py.File('data.h5', 'r')
train_set_x_orig = np.array(train_dataset['X_train'][:]) # your train set features
train_set_y_orig = np.array(train_dataset['y_train'][:]) # your train set labels
test_set_x_orig = np.array(train_dataset['X_test'][:]) # your train set features
test_set_y_orig = np.array(train_dataset['y_test'][:]) # your train set labels
f.close()

print(train_set_x_orig.shape)
print(train_set_y_orig.shape)
 
print(train_set_x_orig.max())
print(train_set_x_orig.min())
 
print(test_set_x_orig.shape)
print(test_set_y_orig.shape)

#测试
plt.imshow(train_set_x_orig[222])
print(train_set_y_orig[222])
'''



