#!/usr/bin/python
from __future__ import print_function
import numpy as np
np.set_printoptions(precision=3)

import time

from six.moves import cPickle as pickle
from six.moves import range

import tensorflow as tf

def reformat(dataset, labels):
    image_size = 28
    num_channels = 1
    num_labels = 10
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/ predictions.shape[0])

def softmax(x):
    res = np.exp(x - max(0.0, np.max(x))) #subtract max value
    #return res / np.sum(res,1)[:,None]
    return res / np.sum(res)

class Layer(object):
    def __init__(self):
        pass

class ConvolutionLayer(Layer):
    def __init__(self,s,c_in,c_out,activation='none'):
        self.W = tf.Variable(tf.truncated_normal([s,s,c_in,c_out],stddev=0.1))
        self.b = tf.Variable(tf.zeros([c_out]))
        self.a = activation
    def ff(self,x):
        res = tf.nn.conv2d(x,self.W,[1,1,1,1],padding='SAME',use_cudnn_on_gpu=True)
        if self.a == 'relu':
            res = tf.nn.relu(res)
        return res + self.b

class DropoutLayer(Layer):
    def __init__(self,p):
        self.p = p
    def ff(self,x):
        return tf.nn.dropout(x,self.p)

class MaxPoolLayer(Layer):
    def __init__(self,ksize,strides):
        self.ksize = ksize #[1,2,2,1]
        self.strides = strides #[1,2,2,1]
        pass
    def ff(self,x):
        return tf.nn.max_pool(x,self.ksize,self.strides,'SAME')

class ActivationLayer(Layer):
    def __init__(self,t):
        self.type = t

    def ff(self,x):
        return tf.nn.relu(x)

class DenseLayer(Layer):

    def __init__(self,shape,activation='none'):
        self.shape = shape
        self.W = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
        self.b = tf.Variable(tf.truncated_normal((shape[-1],),stddev=0.1))
        self.a = activation
        pass

    def ff(self,x):
        if isinstance(x,np.ndarray):
            shape = list(x.shape)
        else:
            shape = x.get_shape().as_list()
        x = tf.reshape(x,[shape[0],reduce(lambda x,y:x*y, shape[1:])]) # batch-flat
        res = tf.matmul(x,self.W)
        if self.a == 'relu':
            res = tf.nn.relu(res)
        return res + self.b

class Net(object):
    def __init__(self):
        self.session = tf.Session()
        #self.optimizer = tf.train.AdagradOptimizer(0.1,0.1)
        self.optimizer = tf.train.RMSPropOptimizer(0.00025,0.95,0.1,1e-10)
        #TODO : Add CLIPPING
        #self.optimizer = tf.train.GradientDescentOptimizer(0.0)
        self.L = [] # Layers

    def setup(self,batch_size,x_shape,y_shape):

        self.tf_X = tf.placeholder(tf.float32,shape=((batch_size,) + x_shape))
        self.tf_Y = tf.placeholder(tf.float32,shape=((batch_size,) + y_shape))

        self.prediction = self._predict(self.tf_X) # estimate
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.prediction,self.tf_Y))
        self.train_step = self.optimizer.minimize(self.loss)

        self.session.run(tf.initialize_all_variables())

    def train(self,X,Y,batch_size,num_steps):
        #vs = ([v for v in tf.all_variables() if 'RMS' in v.name])
        #self.session.run(tf.initialize_variables(vs))
        #self.session.run(tf.initialize_variables([self.optimizer.get_slot(self.loss, name) for name in self.optimizer.get_slot_names()]))

        for step in range(num_steps):
            indices = np.random.randint(Y.shape[0],size = batch_size)
            batch_X = X[indices,:,:,:]
            batch_Y = Y[indices,:]
            feed_dict = {self.tf_X : batch_X, self.tf_Y : batch_Y}
            _,l,pred = self.session.run([self.train_step,self.loss,self.prediction],feed_dict = feed_dict)
            if (step % 100 == 0):
                print('Predictions : ', softmax(pred[0]))
                print('Target : ', batch_Y[0])
                print('Minibatch self.loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % accuracy(pred, batch_Y))

    def _predict(self,X):
        #print('x', X)
        for l in self.L:
            #print(':)')
            X = l.ff(X)
        #print('pred')
        return X

    def predict(self,X):
        return self._predict(X).eval(session = self.session)

    def append(self,l):
        self.L.append(l)

# Activation before or after bias??

# TEST MNIST

if __name__ == "__main__":

    # LOAD DATA
    pickle_file = '../../../DeepLearning/notMNIST.pickle'

    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save  # hint to help gc free up memory
        print('Training set', train_dataset.shape, train_labels.shape)
        print('Validation set', valid_dataset.shape, valid_labels.shape)
        print('Test set', test_dataset.shape, test_labels.shape)

    train_dataset, train_labels = reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)


    batch_size = 16
    patch_size = 5

    num_hidden = 64
    num_labels = 10

    input_shape = (28,28,1)

    output_shape = (10,)

    graph = tf.Graph()

    net = Net()
    net.append(ConvolutionLayer(patch_size,1,16,activation='relu')) #depth : 3 -> 16
    net.append(MaxPoolLayer([1,2,2,1],[1,2,2,1]))
    net.append(ConvolutionLayer(patch_size,16,16,activation='relu')) #depth : 16 -> 16
    net.append(MaxPoolLayer([1,2,2,1],[1,2,2,1]))
    net.append(DropoutLayer(0.5))
    net.append(DenseLayer((28 // 4 * 28 // 4 * 16, num_hidden), activation='relu'))
    net.append(DenseLayer((num_hidden,num_labels),activation='none'))
    net.setup()
    
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    net.train(train_dataset,train_labels, batch_size, 2000)
    net.train(train_dataset,train_labels, batch_size, 2000)
    net.train(train_dataset,train_labels, batch_size, 2000)
    net.train(train_dataset,train_labels, batch_size, 2000)
    net.train(train_dataset,train_labels, batch_size, 2000)

    print('Test accuracy: %.1f%%' % accuracy(net.predict(tf_test_dataset), test_labels))
