import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf

a = tf.placeholder(tf.float32, shape=[3])

b = tf.constant([4,5,6], tf.float32)

c = tf.add(a, b)

with tf.Session() as sess:
	print(sess.run(c,{a:[1,2,3]}))

	# writer = tf.summary.FileWriter('./graphs', sess.graph)
	# sess.run(init)
	# writer.close()