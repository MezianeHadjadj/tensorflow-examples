import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("/tmp/data/", one_hot=True)


alpha = 0.08
num_steps = 500
batch_size = 100

# Network params
n_hidden_1 = 256
n_hidden_2 = 256
num_input = 784
num_classes = 10

# input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])


def neural_net(x):
    weights = {
        'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([num_classes]))
    }
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

logits = neural_net(X)
prediction = tf.nn.softmax(logits)

cost_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=alpha)
train_op = optimizer.minimize(cost_op)


correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


with tf.Session() as s:
    s.run(tf.global_variables_initializer())

    for step in range(1, num_steps+1):
        batch_x, batch_y = data.train.next_batch(batch_size)
        s.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        # Calculate batch loss and accuracy
        cost, acc = s.run([cost_op, accuracy], feed_dict={X: batch_x,
                                                             Y: batch_y})
        print("Step " + str(step) + ", Minibatch cost= " + \
              "{:.4f}".format(cost) + ", Training Accuracy= " + \
              "{:.3f}".format(acc))


    # Run on test data
    print("Testing Accuracy:", s.run(accuracy, feed_dict={X: data.test.images, Y: data.test.labels}))
