#Wilson Burchenal
#Stanford tf tutorial Assignment 1

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

#Basic model parameters
learning_rate = 0.5
batch_size = 128
n_epochs = 25

#Read in data from MNIST
mnist = input_data.read_data_sets('/data/mnist', one_hot=True) 

#Create placeholders for data tensors
x = tf.placeholder(dtype=tf.float32, shape=[batch_size, 784], name="image")
y = tf.placeholder(dtype=tf.float32, shape=[batch_size, 10], name="labels")

#Define weights, bias
w = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.01), name="weights")
b = tf.Variable(tf.zeros(shape=[1,10]), name="bias")

#Create model
logits = tf.matmul(x,w) + b

#Define loss
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y, name='loss')
loss = tf.reduce_mean(entropy)

#Setup model to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


with tf.Session() as sess:

	#Train the model
	start_time=time.time()
	sess.run(tf.global_variables_initializer())	
	n_batches = int(mnist.train.num_examples/batch_size)

	for i in range(n_epochs):
		total_loss = 0
		for _ in range(n_batches):
			X_batch, Y_batch = mnist.train.next_batch(batch_size)
			
			_, loss_batch = sess.run([optimizer, loss], feed_dict={x: X_batch, y: Y_batch})

			total_loss += loss_batch

		print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))

	print('Total time: {0} seconds'.format(time.time() - start_time))

	print('Optimization Finished!') # should be around 0.35 after 25 epochs


	#Test the model
	preds = tf.nn.softmax(logits)
	correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) # need numpy.count_nonzero(boolarr) :(

	n_batches = int(mnist.test.num_examples/batch_size)
	total_correct_preds = 0

	for i in range(n_batches):
		X_batch, Y_batch = mnist.test.next_batch(batch_size)
		accuracy_batch = sess.run([accuracy], feed_dict={x: X_batch, y:Y_batch}) 
		total_correct_preds += sum(accuracy_batch)
	
	print('Accuracy {0}'.format(total_correct_preds/mnist.test.num_examples))
