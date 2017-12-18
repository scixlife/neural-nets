"""
mnist_sgd
~~~~~~~~~

A classifier program for recognizing handwritten digits from the MNIST
data set, using a neural network classifier with stochastic gradient descent."""

"""
Let's start by loading in the MNIST data using a little helper program, mnist_loader.py
"""
import mnist_loader
train, val, test = mnist_loader.load_data_wrapper()

# alternative code to read MNIST data
"""
import gzip, pickle
ff = gzip.open('../data/mnist.pkl.gz','rb')
u = pickle._Unpickler( ff )
u.encoding = 'latin1'
train, val, test = u.load()
ff.close()
"""

"""
After loading the MNIST data, we'll set up a Network with 30 hidden neurons 
using the class ``network``
"""
import network
net = network.Network([784, 30, 10])

"""
Finally, we'll use stochastic gradient descent 
to learn from the MNIST training data 
over 30 epochs, 
with a mini-batch size of 10, 
and a learning rate of η=3.0
"""
net.SGD(train, 30, 10, 3.0, test_data=test)