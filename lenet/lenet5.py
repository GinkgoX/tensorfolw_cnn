import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#load data sets
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

#create session
sess = tf.InteractiveSession()

#define weight function
'''
	description: to initialize weight variable
	Args:	shape [5, 5, 1, 32] #note: conv kernal = 5*5 , color channel = 1, numers of kernal = 32
	Returns:	trunacted_normal distribution of weight variable with stddev = 0.1
'''
def weight_variable(shape):
	init = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(init)

#define biases function
'''
	description: to initialize biases variable
	Args:	shape [5, 5, 1, 32] #note: conv kernal = 5*5 , color channel = 1, numers of kernal = 32
	Returns:	constant biases variable equals 0.1
'''
def biases_variable(shape):
	init = tf.constant(0.1, shape = shape)
	return tf.Variable(init)

#define conv layer
def conv2d(x, w):
	return tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'SAME')

#define max pooling layer
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

#define parameters
n_input = 784
batch_size = 50
training_batch = 100

#define input place holder
x = tf.placeholder(tf.float32, [None, n_input])
y_true = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

#define layer conv_1
w_conv_1 = weight_variable([5, 5, 1, 32])
b_biase_1 = biases_variable([32])
h_conv_1 = tf.nn.relu(conv2d(x_image, w_conv_1) + b_biase_1)
h_pool_1 = max_pool_2x2(h_conv_1)

#define layer cov_2
w_conv_2 = weight_variable([5, 5, 32, 64])
b_biase_2 = biases_variable([64])
h_conv_2 = tf.nn.relu(conv2d(h_pool_1, w_conv_2) + b_biase_2)
h_pool_2 = max_pool_2x2(h_conv_2)

#define full connection layer 1
w_fc_1 = weight_variable([7*7*64, 1024])
b_fc_1 = biases_variable([1024])
flatten_1 = tf.reshape(h_pool_2, [-1, 7*7*64])
h_fc_1 = tf.nn.relu(tf.add(tf.matmul(flatten_1, w_fc_1), b_fc_1))

#define dropout layer
keep_prob = tf.placeholder(tf.float32)
fc_1_dropout = tf.nn.dropout(h_fc_1, keep_prob)

#define full connection layer 2
w_fc_2 = weight_variable([1024, 10])
b_fc_2 = biases_variable([10])
y_pred = tf.nn.softmax(tf.matmul(fc_1_dropout, w_fc_2) + b_fc_2)

#define cost function and optimizer
cost = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y_pred), reduction_indices = [1]))
optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)

#calculate accuracy
correct_predition = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predition, tf.float32))

#build up session
tf.global_variables_initializer().run()
#total_batch = int(mnist.train.num_examples / batch_size)
for i in range(10000):
	batch_xs, batch_ys = mnist.train.next_batch(batch_size)
	if i % 200 == 0:
		train_accuracy = accuracy.eval(feed_dict = {x:batch_xs, y_true:batch_ys, keep_prob:1.0})
		print("step : %d"%(i), " accuracy  = ", "{:.9f}".format(train_accuracy))
	optimizer.run(feed_dict = {x:batch_xs, y_true:batch_ys, keep_prob:0.5})

#test model
print("test accuracy : %g"%accuracy.eval(feed_dict = {x:mnist.test.images, y_true:mnist.test.labels, keep_prob:1.0}))

