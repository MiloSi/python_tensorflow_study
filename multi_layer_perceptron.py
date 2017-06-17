import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
print("Current TF Version Is [%s]" % (tf.__version__))
print("Package Loaded")


mnist = input_data.read_data_sets('data/', one_hot = True)
n_hidden_1 = 256
n_hidden_2 = 128
n_hidden_3 = 64
n_input = 784
n_classes = 10

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

stddev = 0.1
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=stddev)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=stddev)),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], stddev=stddev)),
    'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes], stddev=stddev))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
print("Network Ready")

def multilayer_perceptron(_x, _weights, _biases):
    _layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_x, _weights['h1']), _biases['b1']))
    _layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(_layer_1, _weights['h2']), _biases['b2']))
    _layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(_layer_2, _weights['h3']), _biases['b3']))
    return (tf.matmul(_layer_3, _weights['out']) +_biases['out'])

pred = multilayer_perceptron(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits =pred))
optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accr = tf.reduce_mean(tf.cast(corr, "float"))

#init = tf.global_variables_initializer()
init = tf.initialize_all_variables()
#init = tf.global_norm()
init
print("Function Ready")

training_epoch = 20
batch_size = 100
display_step = 4

sess = tf.Session()
sess.run(init)

for epoch in range(training_epoch):
    avg_cost =  0.
    total_batch = int(mnist.train.num_examples/batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys =  mnist.train.next_batch(batch_size)
        feeds = {x : batch_xs, y : batch_ys}
        sess.run(optm, feed_dict= feeds)
        avg_cost += sess.run(cost, feed_dict=feeds)
    avg_cost = avg_cost/ total_batch
    if(epoch +1) % display_step == 0:
        print("Epoch: %03d/%03d cost : %.9f" % (epoch, training_epoch, avg_cost))
        feeds = {x : batch_xs, y : batch_ys}
        training_acc =sess.run(accr, feed_dict=feeds)
        print ("Training Accuracy: %.3f" %training_acc)
        feeds = {x : mnist.test.images, y : mnist.test.labels}
        test_acc = sess.run(accr, feed_dict=feeds)
        print("Training Accuracy: %.3f" % training_acc)
print("Optimization Finished")