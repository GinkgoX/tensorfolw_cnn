import tensorflow as tf
import matplotlib.pyplot as plt
import time
import numpy as np
import cifar10, cifar10_input

#define patameters
max_steps = 8000
batch_size = 512
n_fc_1 = 128
n_fc_2 = 64
display_step = 200
data_dir = 'cifar-10-binary/cifar-10-batches-bin'

#define L2 regularization for weight loss
def l2_weight_loss(shape, stddev, w_1):
	'''
		description: to adopt L2 regularization for weight to prevent over-fitting
		Args:	shape:
				stddev: standard deviation
				w_1: weight coeffient
		Returns:
				weight: the regularized weight coeffient
	'''
	weight = tf.Variable(tf.truncated_normal(shape, stddev))
	if w_1 is not None:
		weight_loss = tf.multiply(tf.nn.l2_loss(weight), w_1, name='weight_loss')
		tf.add_to_collection('losses', weight_loss)
	return weight

#define weight initializer
def weight_init(shape, stddev):
	'''
		description: to initialize weight values
		Args:	shape:
				stddev: standard deviation
		Returns:
				weight: weight values
	'''
	return tf.Variable(tf.truncated_normal(shape, stddev))

#define biases initializer
def biases_init(shape):
	'''
		description: to initialize biases values
		Args:	shape:
		Returns:	biases values
	'''
	return tf.Variable(tf.random_normal(shape))

#define conv layer
def conv2d(x_image, weight):
	'''
		description: to excute conv operation
		Args:	x_image: input data
				weight: weight values
		Returns: conv result
	'''
	return tf.nn.conv2d(x_image, weight, strides = [1, 1, 1, 1], padding = 'SAME')

#define max pooling layer
def max_pool(x_image):
	'''
		description: to adopt max pooling method
		Args:	x_image: input data
		Returns:	max pooling result
	'''
	return tf.nn.max_pool(x_image, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME')

def lrn_norm(x_image):
	'''
		description : to adopt LRN(local response neurial) normalization
		Args:	x_image: input data
		Returns:	the lrn normalized result
	'''
	return tf.nn.lrn(x_image, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)

#image enhancement processing for trainning data sets
train_images, train_labels = cifar10_input.distorted_inputs(batch_size = batch_size, data_dir = data_dir)

#define input placeholder
x_images = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
x_labels = tf.placeholder(tf.float32, [batch_size])

#define layer 1
w_1 = weight_init([5, 5, 3, 32], 0.05)
b_1 = biases_init([32])
conv_1 = tf.nn.relu(conv2d(x_images, w_1) + b_1)
pool_1 = max_pool(conv_1)
lrn_1 = lrn_norm(pool_1)

#define layer 2
w_2 = weight_init([5, 5, 32, 32], 0.05)
b_2 = biases_init([32])
conv_2 = tf.nn.relu(conv2d(lrn_1, w_2) + b_2)
pool_2 = max_pool(conv_2)

#define flatten layer
re_shape = tf.reshape(pool_2, [batch_size, -1])
n_input = re_shape.get_shape()[1].value

#define full connection 1
w_3 = l2_weight_loss([n_input, n_fc_1], 0.05, w_1 = 0.001)
b_3 = biases_init([n_fc_1])
fc_1 = tf.nn.relu(tf.matmul(re_shape, w_3) + b_3)

#define full connection 2
w_4 = l2_weight_loss([n_fc_1, n_fc_2], 0.05, w_1 = 0.003)
b_4 = biases_init([n_fc_2])
fc_2 = tf.nn.relu(tf.matmul(fc_1, w_4) + b_4)

#define output layer
w_5 = weight_init([n_fc_2, 10], 1.0 / 96.0)
b_5 = biases_init([10])
logits = tf.add(tf.matmul(fc_2, w_5), b_5)
y_pred = tf.nn.softmax(logits)

#define loss function
x_labels = tf.cast(x_labels, tf.int32)
cross_enerty = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = x_labels, name = 'cross_enerty_pre_example')
losses = tf.reduce_mean(cross_enerty, name = 'cross_enerty')
tf.add_to_collection('losses', losses)
loss = tf.add_n(tf.get_collection('losses'), name = 'total_loss')

#define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(loss)

#display test result
test_images, test_labels = cifar10_input.inputs(batch_size = batch_size, data_dir = data_dir, eval_data = True)

#claculate trainning accuracy
def accuracy(test_labels, y_pred):
	'''
		description: to claculate the trainning accuracy
		Args:	test_labels: test labels
				y_pred: conv output value
		Returns:
				accuracy of trainning result
	'''
	test_labels = tf.to_int64(test_labels)
	correction_pred = tf.equal(test_labels, tf.argmax(y_pred, 1))
	acc = tf.reduce_mean(tf.cast(correction_pred, tf.float32))
	return acc

#create session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

#create thread to accelaeate processing efficience
tf.train.start_queue_runners(sess = sess)

#start trainning and display
cross_loss = []
for i in range(max_steps):
	start_time = time.time()
	batch_xs, batch_ys = sess.run([train_images, train_labels])
	_, c = sess.run([optimizer, loss], feed_dict = {x_images:batch_xs, x_labels:batch_ys})
	cross_loss.append(c)
	every_epoch_time = time.time() - start_time
	if i % display_step == 0:
		examples_per_sec = batch_size / every_epoch_time
		every_batch_time = float(every_epoch_time)
		print('Epoch : ', '%d'%(i+100), 'loss : ', '{:.5f}'.format(c))
print("optimization finished ! ")

#display loss processing
fig, ax = plt.subplots(figsize = (13, 6))
ax.plot(cross_loss)
plt.grid()
plt.title("trian loss")
plt.show() 

#claculate test accuracy
for i in range(10):
	test_acc = []
	batch_xs, batch_ys = sess.run([test_images, test_labels])
	batch_y_pred = sess.run(y_pred, feed_dict = {x_images:batch_xs})
	test_accuracy = accuracy(batch_ys, batch_y_pred)
	acc = sess.run(test_accuracy, feed_dict = {x_images:batch_xs})
	test_acc.append(acc)
	print("test accuracy : ", acc)
print("mean accuracy : ", np.mean(test_acc))

