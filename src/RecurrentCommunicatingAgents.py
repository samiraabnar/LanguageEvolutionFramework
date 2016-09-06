import theano
import theano.tensor as T
import numpy as np
from scipy.spatial import distance


import sys
sys.path.append('../../')

from ReinforcedFeedForwardNetwork.src.VGG_16 import *

class ReinforcedFeedForwardNeuralNetwork_Listener(object):
    def __init__(self,dims):
        self.dimensions = dims
        self.number_of_layers = len(dims)
        self.learning_rate = 0.1



    def define_network(self):

        X = T.matrix('input')
        Y = T.vector('output')


        W = []
        for i in np.arange(self.number_of_layers - 1):
            w = self.init_weights((self.dimensions[i],self.dimensions[i+1]))
            W.append(w)


        layers_activations = []
        layers_activations.append(X)
        for i in np.arange(1,self.number_of_layers):
            activation = theano.shared(np.asarray(np.zeros(self.dimensions)))
            layers_activations.append(activation)

        for i in np.arange(1,self.number_of_layers - 1):
            layers_activations[i] = T.nnet.sigmoid(T.dot(layers_activations[i-1], W[i-1]))

        layers_activations[-1] = T.dot(layers_activations[-2], W[-1])

        output = layers_activations[-1]

        self.forward_step = theano.function([X],[output])

        cost = T.mean((output - Y) ** 2)
        grads = T.grad(cost,W)
        updates = [(param_i, param_i - self.learning_rate * grad_i) for param_i, grad_i in zip(W, grads)]
        self.backprop_update = theano.function([X,Y],[],updates=updates)


    def reinforce_weights(self,W):
        pass



    def init_weights(self,shape):
        """ Weight initialization """
        weights = np.asarray(np.random.randn(*shape), dtype=theano.config.floatX)
        return theano.shared(weights)


class ReinforcedFeedForwardNeuralNetwork_Talker(object):
    def __init__(self,dims):
        self.dimensions = dims
        self.number_of_layers = len(dims)
        self.learning_rate = 0.1



    def define_network(self):

        X = T.matrix('input')
        Y = T.vector('output')


        W = []
        for i in np.arange(self.number_of_layers - 1):
            w = self.init_weights((self.dimensions[i],self.dimensions[i+1]))
            W.append(w)


        layers_activations = []
        layers_activations.append(X)
        for i in np.arange(1,self.number_of_layers):
            activation = theano.shared(np.asarray(np.zeros(self.dimensions)))
            layers_activations.append(activation)

        for i in np.arange(1,self.number_of_layers):
            layers_activations[i] = T.nnet.sigmoid(T.dot(layers_activations[i-1], W[i-1]))


        output = layers_activations[-1]

        self.forward_step = theano.function([X],[output])

        cost = T.mean((output - Y) ** 2)
        grads = T.grad(cost,W)
        updates = [(param_i, param_i - self.learning_rate * grad_i) for param_i, grad_i in zip(W, grads)]
        self.backprop_update = theano.function([X,Y],[],updates=updates)


    def reinforce_weights(self,W):
        pass



    def init_weights(self,shape):
        """ Weight initialization """
        weights = np.asarray(np.random.randn(*shape), dtype=theano.config.floatX)
        return theano.shared(weights)



Alpha = ['A', 'B'] #, 'C', 'D', 'E']
def get_string(vec):
    str = " "
    for i in np.arange(len(vec)):
        if(vec[i] == 1):
            str += Alpha[i]

    return str

if __name__ == '__main__':
    rfnn_talker = ReinforcedFeedForwardNeuralNetwork_Talker([4096,512,2])
    rfnn_talker.define_network()

    rfnn_listener = ReinforcedFeedForwardNeuralNetwork_Listener([2, 4096*2, 4096])
    rfnn_listener.define_network()


    images,thumb_images = load_images_from_folder("shapes")
    vgg = VGG_16('vgg16_weights.h5')
    vgg.model.compile(optimizer='adam', loss='categorical_crossentropy')


    for k in np.arange(5000):
        i = np.random.randint(4)#len(images))
        j = np.random.randint(4)#len(images))

        while j == i:
            j = np.random.randint(4)#len(images))

        rep1 = vgg.get_representation(images[i])
        rep2 = vgg.get_representation(images[j])


        """
        d = np.zeros((1,5),dtype='float32')
        d[0][i] = 1.0
        """

        description0 = rfnn_talker.forward_step(rep1)#[d]#
        description1 = description0[0]
        description = np.transpose(description1)
        recons_rep = rfnn_listener.forward_step(np.round(description1))

        dist1 = distance.euclidean(recons_rep,rep1)
        dist2 = distance.euclidean(recons_rep,rep2)

        selected = rep1
        state = "succeed!"
        if dist1 > dist2:
            selected = rep2
            state = "Failed!"


        print(get_string(np.round(description1[0]))+" "+str(i))
        print(state)
        rfnn_talker.backprop_update(selected,np.round(description0[0][0]))
        rfnn_listener.backprop_update(np.round(description1),rep1[0])

