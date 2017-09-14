import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
from numpy import genfromtxt
from sklearn.datasets import load_boston


#read a file csv or  tsv...
def read_dataset(filePath,delimiter=','):
    return genfromtxt(filePath, delimiter=delimiter)

# read our data (features, labels) and transfer it to numpy array
def read_boston_data():
    boston = load_boston()
    features = np.array(boston.data)
    labels = np.array(boston.target)
    print features;
    return features, labels

# normalize our data to convert all features to same scale
def feature_normalize(dataset):
    mu = np.mean(dataset,axis=0)
    sigma = np.std(dataset,axis=0)
    return (dataset - mu)/sigma

# append bias 1's to our normalized features
def append_bias_reshape(features,labels):
    n_training_samples = features.shape[0]
    n_dim = features.shape[1]
    f = np.reshape(np.c_[np.ones(n_training_samples),features],[n_training_samples,n_dim + 1])
    print features
    print  "----"
    l = np.reshape(labels,[n_training_samples,1])
    print f
    return f, l

#load data and normalize and append bias
features,labels = read_boston_data()
normalized_features = feature_normalize(features)
f, l = append_bias_reshape(normalized_features,labels)
n_dim = f.shape[1]

# separate training data from test data
rnd_indices = np.random.rand(len(f)) < 0.80
train_x = f[rnd_indices]
train_y = l[rnd_indices]
test_x = f[~rnd_indices]
test_y = l[~rnd_indices]

#define basic params
learning_rate = 0.01
training_epochs = 1000
cost_history = np.empty(shape=[1],dtype=float)
#declare structure with tf way
X = tf.placeholder(tf.float32,[None,n_dim])
Y = tf.placeholder(tf.float32,[None,1])
W = tf.Variable(tf.ones([n_dim,1]))
#initialize all variables we defined (graph)
init = tf.global_variables_initializer()

#Lineare regression with TF: line 1 multiply features by weights, line 2 calculate cost function j, line 3 update wights using gradient descent

y_ = tf.matmul(X, W)
cost = tf.reduce_mean(tf.square(y_ - Y))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.Session() as s:
    s.run(init)
    #print s.run(normalized_features)
    #print s.run(f)
    for epoch in range(training_epochs):
        s.run(training_step,feed_dict={X:train_x,Y:train_y})
        cost_history = np.append(cost_history,s.run(cost,feed_dict={X: train_x,Y: train_y}))
    plt.plot(range(len(cost_history)),cost_history)
    plt.axis([0,training_epochs,0,np.max(cost_history)])
    plt.show()

    # predict on test dataset and calculate mean square error to see how much is efficient the training
    pred_y = s.run(y_, feed_dict={X: test_x})
    mse = tf.reduce_mean(tf.square(pred_y - test_y))
    print("MSE: %.4f" % s.run(mse))

    fig, ax = plt.subplots()
    ax.scatter(test_y, pred_y)
    ax.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=3)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()
