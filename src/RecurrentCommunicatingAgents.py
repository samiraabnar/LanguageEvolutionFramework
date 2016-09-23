import theano
import theano.tensor as T
import numpy as np
from scipy.spatial import distance
import sklearn

import sys
sys.path.append('../../')

from LanguageEvolutionFramework.src.Vision.VGG_16 import *

class LSTM_Listener(object):
    def __init__(self,input_dim,output_dim,outer_output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.outer_output_dim = outer_output_dim
        self.learning_rate = 0.1
        self.random_state = np.random.RandomState(23455)
        self.initial_hiddens = [None,dict(initial=T.zeros(self.output_dim,dtype=theano.config.floatX)),
                          dict(initial=T.zeros(self.output_dim,dtype=theano.config.floatX))
                          , None, None, None
                          ]


    def init_lstm_weights(self):
        U_input = np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.input_dim + self.output_dim)),
                                     np.sqrt(6.0 / (self.input_dim + self.output_dim)),
                                     (self.output_dim, self.input_dim))
            , dtype=theano.config.floatX)

        U_forget = np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.input_dim + self.output_dim)),
                                     np.sqrt(6.0 / (self.input_dim + self.output_dim)),
                                     (self.output_dim, self.input_dim))
            , dtype=theano.config.floatX)

        U_output = np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.input_dim + self.output_dim)),
                                     np.sqrt(6.0 / (self.input_dim + self.output_dim)),
                                     (self.output_dim, self.input_dim))
            , dtype=theano.config.floatX)

        W_input = np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     (self.output_dim, self.output_dim))
            , dtype=theano.config.floatX)

        W_forget = np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     (self.output_dim, self.output_dim))
            , dtype=theano.config.floatX)

        W_output = np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     (self.output_dim, self.output_dim))
            , dtype=theano.config.floatX)

        U = np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.input_dim + self.output_dim)),
                                     np.sqrt(6.0 / (self.input_dim + self.output_dim)),
                                     (self.output_dim, self.input_dim))
            , dtype=theano.config.floatX)

        W = np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     (self.output_dim, self.output_dim))
            , dtype=theano.config.floatX)



        self.W = theano.shared(value=W, name="W" , borrow="True")
        self.U = theano.shared(value=U, name="U" , borrow="True")

        self.W_input = theano.shared(value=W_input, name="W_input" , borrow="True")
        self.U_input = theano.shared(value=U_input, name="U_input" , borrow="True")

        self.W_output = theano.shared(value=W_output, name="W_output" , borrow="True")
        self.U_output = theano.shared(value=U_output, name="U_output" , borrow="True")

        self.W_forget = theano.shared(value=W_forget, name="W_forget" , borrow="True")
        self.U_forget = theano.shared(value=U_forget, name="U_forget" , borrow="True")


        U_input_2 = np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     (self.output_dim, self.output_dim))
            , dtype=theano.config.floatX)

        U_forget_2 = np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     (self.output_dim, self.output_dim))
            , dtype=theano.config.floatX)

        U_output_2 = np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     (self.output_dim, self.output_dim))
            , dtype=theano.config.floatX)

        W_input_2 = np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     (self.output_dim, self.output_dim))
            , dtype=theano.config.floatX)

        W_forget_2 = np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     (self.output_dim, self.output_dim))
            , dtype=theano.config.floatX)

        W_output_2 = np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     (self.output_dim, self.output_dim))
            , dtype=theano.config.floatX)

        U_2 = np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.input_dim + self.output_dim)),
                                     np.sqrt(6.0 / (self.input_dim + self.output_dim)),
                                     (self.output_dim, self.output_dim))
            , dtype=theano.config.floatX)

        W_2 = np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     (self.output_dim, self.output_dim))
            , dtype=theano.config.floatX)



        self.W_2 = theano.shared(value=W_2, name="W_2" , borrow="True")
        self.U_2 = theano.shared(value=U_2, name="U_2" , borrow="True")

        self.W_input_2 = theano.shared(value=W_input_2, name="W_input_2" , borrow="True")
        self.U_input_2 = theano.shared(value=U_input_2, name="U_input_2" , borrow="True")

        self.W_output_2 = theano.shared(value=W_output_2, name="W_output_2" , borrow="True")
        self.U_output_2 = theano.shared(value=U_output_2, name="U_output_2" , borrow="True")

        self.W_forget_2 = theano.shared(value=W_forget_2, name="W_forget_2" , borrow="True")
        self.U_forget_2 = theano.shared(value=U_forget_2, name="U_forget_2" , borrow="True")





        O_w = np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     (self.outer_output_dim, self.output_dim))
            , dtype=theano.config.floatX)



        self.O_w = theano.shared(value=O_w, name="O_w" , borrow="True")

        W_in = np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     (4096, self.outer_output_dim))
            , dtype=theano.config.floatX)

        W_out = np.asarray(self.random_state.normal(-np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     (4096, self.outer_output_dim))
            , dtype=theano.config.floatX)

        self.W_in = theano.shared(value=W_in, name="W_in" , borrow="True")
        self.W_out = theano.shared(value=W_out, name="W_out", borrow="True")

        self.params = [self.U_input, self.U_forget, self.U_output, self.W_input, self.W_forget, self.W_output,
                       self.U, self.W,
                       self.U_input_2, self.U_forget_2, self.U_output_2, self.W_input_2, self.W_forget_2, self.W_output_2,
                       self.U_2, self.W_2,
                       self.O_w
                       ]

        self.output_params = [self.W_out]


    def define_network(self):

        X = T.matrix('input')
        Y = T.vector('output')

        H = T.vector('init_state')

        self.init_lstm_weights()

        def forward_step(x_t, prev_state, prev_content,prev_state_2,prev_content_2):
            input_gate = T.nnet.hard_sigmoid(
                T.dot((self.U_input), x_t) + T.dot(self.W_input, prev_state))
            forget_gate = T.nnet.hard_sigmoid(
                T.dot((self.U_forget), x_t) + T.dot(self.W_forget, prev_state))
            output_gate = T.nnet.hard_sigmoid(
                T.dot((self.U_output), x_t) + T.dot(self.W_output, prev_state))

            stabilized_input = T.tanh(T.dot((self.U), x_t) + T.dot(self.W, prev_state))
            c = forget_gate * prev_content + input_gate * stabilized_input
            s = output_gate * T.tanh(c)

            input_gate2 = T.nnet.hard_sigmoid(
                T.dot((self.U_input_2), s) + T.dot(self.W_input_2, prev_state_2))
            forget_gate2 = T.nnet.hard_sigmoid(
                T.dot((self.U_forget_2), s) + T.dot(self.W_forget_2, prev_state_2))
            output_gate2 = T.nnet.hard_sigmoid(
                T.dot((self.U_output_2), s) + T.dot(self.W_output_2, prev_state_2))

            stabilized_input2 = T.tanh(T.dot((self.U_2), s) + T.dot(self.W_2, prev_state_2))
            c2 = forget_gate2 * prev_content_2 + input_gate2 * stabilized_input2
            s2 = output_gate2 * T.tanh(c2)



            o = T.nnet.softmax(T.dot(self.O_w, s2))[0]

            return [o, s, c, s2, c2, input_gate, forget_gate, output_gate]



        """self.initial_hiddens = [None,dict(initial=H),
                                dict(initial=T.zeros(self.output_dim,dtype=theano.config.floatX))
                                , None, None, None
                                ]

        """

        [self.output,self.hidden_state,self.memory_content, _, _,self.input_gate, self.forget_gate,
         self.output_gate] , updates = theano.scan(
            forward_step,
            sequences=[X],
            truncate_gradient=-1,
            outputs_info= [None,dict(initial=T.zeros(self.output_dim,dtype=theano.config.floatX)),
                               dict(initial=T.zeros(self.output_dim,dtype=theano.config.floatX)),
                               dict(initial=T.zeros(self.output_dim,dtype=theano.config.floatX)),
                               dict(initial=T.zeros(self.output_dim,dtype=theano.config.floatX))
                               , None, None, None
                               ])




        output = T.mean(self.output,axis=0)#T.nnet.softmax(T.dot(self.W_out,T.sum(self.output.T,axis=1)))[0]
        self.predict = theano.function([X],[output])

        params = self.params #+ self.output_params
        cost = T.sum(T.nnet.categorical_crossentropy(output,Y))# T.sum(distance.euclidean(output,Y))  #
        grads = T.grad(cost,params)
        updates = [(param_i, param_i - self.learning_rate * grad_i) for param_i, grad_i in zip(params, grads)]
        self.backprop_update = theano.function([X,Y],[cost],updates=updates)
        p_updates = [(param_i, param_i + self.learning_rate * param_i + param_i*np.random.rand()) for param_i in params]
        self.positive_reinforcement = theano.function([],[],updates=p_updates)
        n_updates = [(param_i, param_i - self.learning_rate * param_i + param_i*np.random.rand()) for param_i in params]
        self.negetive_reinforcement = theano.function([],[],updates=n_updates)




class LSTM_Talker(object):
    def __init__(self, input_dim, output_dim, outer_output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.outer_output_dim = outer_output_dim
        self.learning_rate = 0.03
        self.random_state = np.random.RandomState(23455)
        self.initial_hiddens = [None, dict(initial=T.zeros(self.output_dim, dtype=theano.config.floatX)),
                                dict(initial=T.zeros(self.output_dim, dtype=theano.config.floatX))
            , None, None, None
                                ]

    def init_lstm_weights(self):
        U_input = np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.input_dim + self.output_dim)),
                                     np.sqrt(6.0 / (self.input_dim + self.output_dim)),
                                     (self.output_dim, self.input_dim))
            , dtype=theano.config.floatX)

        U_forget = np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.input_dim + self.output_dim)),
                                     np.sqrt(6.0 / (self.input_dim + self.output_dim)),
                                     (self.output_dim, self.input_dim))
            , dtype=theano.config.floatX)

        U_output = np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.input_dim + self.output_dim)),
                                     np.sqrt(6.0 / (self.input_dim + self.output_dim)),
                                     (self.output_dim, self.input_dim))
            , dtype=theano.config.floatX)

        W_input = np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     (self.output_dim, self.output_dim))
            , dtype=theano.config.floatX)

        W_forget = np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     (self.output_dim, self.output_dim))
            , dtype=theano.config.floatX)

        W_output = np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     (self.output_dim, self.output_dim))
            , dtype=theano.config.floatX)

        U = np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.input_dim + self.output_dim)),
                                     np.sqrt(6.0 / (self.input_dim + self.output_dim)),
                                     (self.output_dim, self.input_dim))
            , dtype=theano.config.floatX)

        W = np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     (self.output_dim, self.output_dim))
            , dtype=theano.config.floatX)

        self.W = theano.shared(value=W, name="W", borrow="True")
        self.U = theano.shared(value=U, name="U", borrow="True")

        self.W_input = theano.shared(value=W_input, name="W_input", borrow="True")
        self.U_input = theano.shared(value=U_input, name="U_input", borrow="True")

        self.W_output = theano.shared(value=W_output, name="W_output", borrow="True")
        self.U_output = theano.shared(value=U_output, name="U_output", borrow="True")

        self.W_forget = theano.shared(value=W_forget, name="W_forget", borrow="True")
        self.U_forget = theano.shared(value=U_forget, name="U_forget", borrow="True")

        O_w = np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     (self.outer_output_dim, self.output_dim))
            , dtype=theano.config.floatX)

        self.O_w = theano.shared(value=O_w, name="O_w", borrow="True")

        W_in = np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     (4096, self.outer_output_dim))
            , dtype=theano.config.floatX)

        W_out = np.asarray(self.random_state.normal(-np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                                    np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                                    (4096, self.outer_output_dim))
                           , dtype=theano.config.floatX)

        self.W_in = theano.shared(value=W_in, name="W_in", borrow="True")
        self.W_out = theano.shared(value=W_out, name="W_out", borrow="True")

        self.params = [self.U_input, self.U_forget, self.U_output, self.W_input, self.W_forget, self.W_output,
                       self.U, self.W, self.W_out
                       ]

        self.output_params = [self.O_w]

    def define_network(self):
        X = T.matrix('input')
        Y = T.vector('output')

        self.init_lstm_weights()

        def forward_step(x_t, prev_state, prev_content):
            input_gate = T.nnet.hard_sigmoid(
                T.dot((self.U_input), x_t) + T.dot(self.W_input, prev_state))
            forget_gate = T.nnet.hard_sigmoid(
                T.dot((self.U_forget), x_t) + T.dot(self.W_forget, prev_state))
            output_gate = T.nnet.hard_sigmoid(
                T.dot((self.U_output), x_t) + T.dot(self.W_output, prev_state))

            stabilized_input = T.tanh(T.dot((self.U), x_t) + T.dot(self.W, prev_state))
            c = forget_gate * prev_content + input_gate * stabilized_input
            s = output_gate * T.tanh(c)

            o = T.dot(self.O_w, s)

            return [o, s, c, input_gate, forget_gate, output_gate]

        [self.output, self.hidden_state, self.memory_content, self.input_gate, self.forget_gate,
         self.output_gate], updates = theano.scan(
            forward_step,
            sequences=[input],
            truncate_gradient=-1,
            outputs_info=self.initial_hiddens)

        output = T.dot(self.W_out, self.output.T)
        self.predict = theano.function([X], [output])

        params = self.params + self.output_params
        cost = T.mean((output - Y) ** 2)
        grads = T.grad(cost, params)
        updates = [(param_i, param_i - self.learning_rate * grad_i) for param_i, grad_i in zip(params, grads)]
        self.backprop_update = theano.function([X, Y], [], updates=updates)


Alpha = ['A', 'B', 'C', 'D', 'E']
def get_string(list_of_vec):
    str = " "
    for i in np.arange(len(list_of_vec)):
            str += Alpha[np.argmax(list_of_vec[i])]

    return str

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

if __name__ == '__main__':
    #rfnn_talker = AttendedLSTM(input_dim=5, output_dim=5, number_of_layers=1,
    #                               hidden_dims=[512], dropout_p=0.0, learning_rate=0.01) # ReinforcedFeedForwardNeuralNetwork_Talker([4096,512,2])
    #rfnn_talker.build_model()

    lstm_listener = LSTM_Listener(input_dim=20, output_dim=1000, outer_output_dim=4096)#ReinforcedFeedForwardNeuralNetwork_Listener([2, 4096*2, 4096])
    lstm_listener.define_network()


    images,thumb_images = load_images_from_folder("shapes")
    vgg = VGG_16('vgg16_weights.h5')
    vgg.model.compile(optimizer='adam', loss='categorical_crossentropy')


    y = T.matrix()
    o = T.matrix()
    dist_fun = theano.function([y,o] ,[T.nnet.categorical_crossentropy(y, o)])



    for k in np.arange(1000):
        i = np.random.randint(2)#len(images))
        j = np.random.randint(2)#len(images))





        while j == i:
            j = np.random.randint(2)#len(images))

        rep1 = softmax(vgg.get_representation(images[i])[0])
        rep2 = softmax(vgg.get_representation(images[j])[0])



        d = np.zeros((2,20),dtype='float32')
        d[0][i%2] = 1.0
        d[1][(i+1)%2] = 1.0


        #description0 = rfnn_talker.forward_step(rep1)#[d]#
        #description1 = description0[0]
        #description = np.transpose(description1)
        seq_embedding = lstm_listener.predict(d)

        dist1 = dist_fun(np.asarray([rep1]),np.asarray(seq_embedding))     #distance.euclidean(rep1,seq_embedding)#
        dist2 = dist_fun(np.asarray([rep2]),np.asarray(seq_embedding))     #distance.euclidean(rep2,seq_embedding) #

        selected = rep1
        state = "Succeed!"
        if dist2 < dist1:
                selected = rep2
                state = "Failed!"



        [cost1] = lstm_listener.backprop_update(d,rep1)
        print(get_string(d)+" "+str(i))
        print(str(cost1))
        print(state)
        #rfnn_talker.backprop_update(selected,np.round(description0[0][0]))


