import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#load data sets
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

#define input nodes and hidden nodes
n_input = 784
n_hidden_1 = 300

#define parameters
batch_size = 100
training_epochs = 10
display_step = 1

#define input placeholder
x = tf.placeholder(tf.float32, [None, n_input])

#define rates of dropout
keep_prop = tf.placeholder(tf.float32)

#define weights and biases dict
weights = {
	'W_1' : tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev = 0.1)),
	'W_2' : tf.Variable(tf.zeros([n_hidden_1, 10])),
}

biases = {
	'b_1' : tf.Variable(tf.zeros([n_hidden_1])),
	'b_2' : tf.Variable(tf.zeros([10])),
}

#build model
layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['W_1']), biases['b_1']))
layer_2 = tf.nn.dropout(layer_1, keep_prop)
y_pred = tf.nn.softmax(tf.add(tf.matmul(layer_1, weights['W_2']), biases['b_2']))

#define output placeholder
y_true = tf.placeholder(tf.float32, [None, 10])

#define cost function[cross_enerty] and optimizer
cost = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y_pred), reduction_indices = [1]))
optimizer = tf.train.AdagradOptimizer(0.3).minimize(cost)

#create session and initialize variables
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

#train the model
total_batch = int(mnist.train.num_examples / batch_size)
for eoch in range(training_epochs):
	for i in range(total_batch):
		batch_xs, batch_ys = mnist.train.next_batch(batch_size)
		_, c = sess.run([optimizer, cost], feed_dict = {x:batch_xs, y_true:batch_ys, keep_prop:0.75})
	if eoch % display_step == 0:
		print("Eoch : ", '%04d'%(eoch + 1), "Cost = ", '{:.9f}'.format(c))
print("optimization finished ! ")

#calaulate accuracy
correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#test model accuracy
print(accuracy.eval({x:mnist.test.images, y_true:mnist.test.labels, keep_prop:1.0}))
