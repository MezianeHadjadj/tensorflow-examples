import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

h = tf.nn.softmax(tf.matmul(x, W) + b)

y = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(h), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

for _ in range(10):
    batch_xs, batch_ys = mnist.train.next_batch(5)
    print (sess.run(cross_entropy, feed_dict={x: batch_xs, y: batch_ys}))
    print ('------------------')
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

#evaluate

correct_prediction = tf.equal(tf.argmax(h,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
