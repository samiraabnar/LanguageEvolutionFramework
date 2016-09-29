import theano.tensor as T
import theano
import numpy as np
from scipy.spatial import distance


import sys

sys.path.append('../../')

from LanguageEvolutionFramework.src.Vision.VGG_16 import *
from theanolm.matrixfunctions import orthogonal_weight


class CaptionGenerator(object):

    def __init__(self,input_dim,hidden_dim,output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = 0.1
        self.random_state = np.random.RandomState(23455)

        self.EmptyLetter = np.asarray(np.eye(output_dim)[0],dtype='float32')

        self.image_reader = VGG_16('vgg16_weights.h5')
        self.image_reader.model.compile(optimizer='adam', loss='categorical_crossentropy')


    def init_lstm_weights(self):
        U_input = orthogonal_weight(self.hidden_dim, self.input_dim,scale=0.01)
        """np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.output_dim + self.hidden_dim)),
                                     np.sqrt(6.0 / (self.output_dim + self.hidden_dim)),
                                     (self.hidden_dim, self.input_dim))
            , dtype=theano.config.floatX)
        """
        U_forget = orthogonal_weight(self.hidden_dim, self.input_dim,scale=0.01)
        """np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.output_dim + self.hidden_dim)),
                                     np.sqrt(6.0 / (self.output_dim + self.hidden_dim)),
                                     (self.hidden_dim, self.input_dim))
            , dtype=theano.config.floatX)
        """
        U_output = orthogonal_weight(self.hidden_dim, self.input_dim,scale=0.01)
        """np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.output_dim + self.hidden_dim)),
                                     np.sqrt(6.0 / (self.output_dim + self.hidden_dim)),
                                     (self.hidden_dim, self.input_dim))
            , dtype=theano.config.floatX)
        """
        W_input = orthogonal_weight(self.hidden_dim, self.hidden_dim,scale=0.01)
        """np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.hidden_dim + self.hidden_dim)),
                                     np.sqrt(6.0 / (self.hidden_dim + self.hidden_dim)),
                                     (self.hidden_dim, self.hidden_dim))
            , dtype=theano.config.floatX)
        """
        W_forget = orthogonal_weight(self.hidden_dim, self.hidden_dim,scale=0.01)
        """np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.hidden_dim + self.hidden_dim)),
                                     np.sqrt(6.0 / (self.hidden_dim + self.hidden_dim)),
                                     (self.hidden_dim, self.hidden_dim))
            , dtype=theano.config.floatX)
        """
        W_output = orthogonal_weight(self.hidden_dim, self.hidden_dim,scale=0.01)
        """np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.hidden_dim + self.hidden_dim)),
                                     np.sqrt(6.0 / (self.hidden_dim + self.hidden_dim)),
                                     (self.hidden_dim, self.hidden_dim))
            , dtype=theano.config.floatX)
        """
        U = orthogonal_weight(self.hidden_dim, self.input_dim,scale=0.01)
        """np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.output_dim + self.hidden_dim)),
                                     np.sqrt(6.0 / (self.output_dim + self.hidden_dim)),
                                     (self.input_dim, self.input_dim))
            , dtype=theano.config.floatX)
        """
        W = orthogonal_weight(self.hidden_dim, self.hidden_dim,scale=0.01)
        """np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.hidden_dim + self.hidden_dim)),
                                     np.sqrt(6.0 / (self.hidden_dim + self.hidden_dim)),
                                     (self.hidden_dim, self.hidden_dim))
            , dtype=theano.config.floatX)
        """


        self.W = theano.shared(value=W, name="W" , borrow="True")
        self.U = theano.shared(value=U, name="U" , borrow="True")

        self.W_input = theano.shared(value=W_input, name="W_input" , borrow="True")
        self.U_input = theano.shared(value=U_input, name="U_input" , borrow="True")

        self.W_output = theano.shared(value=W_output, name="W_output" , borrow="True")
        self.U_output = theano.shared(value=U_output, name="U_output" , borrow="True")

        self.W_forget = theano.shared(value=W_forget, name="W_forget" , borrow="True")
        self.U_forget = theano.shared(value=U_forget, name="U_forget" , borrow="True")


        U_input_2 = orthogonal_weight(self.hidden_dim, self.hidden_dim)
        """np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.hidden_dim + self.hidden_dim)),
                                     np.sqrt(6.0 / (self.hidden_dim + self.hidden_dim)),
                                     (self.hidden_dim, self.hidden_dim))
            , dtype=theano.config.floatX)
        """
        U_forget_2 = orthogonal_weight(self.hidden_dim, self.hidden_dim,scale=0.01)
        """np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.hidden_dim + self.hidden_dim)),
                                     np.sqrt(6.0 / (self.hidden_dim + self.hidden_dim)),
                                     (self.hidden_dim, self.hidden_dim))
            , dtype=theano.config.floatX)
        """
        U_output_2 = orthogonal_weight(self.hidden_dim, self.hidden_dim,scale=0.01)
        """np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.hidden_dim + self.hidden_dim)),
                                     np.sqrt(6.0 / (self.hidden_dim + self.hidden_dim)),
                                     (self.hidden_dim, self.hidden_dim))
            , dtype=theano.config.floatX)
        """
        W_input_2 = orthogonal_weight(self.hidden_dim, self.hidden_dim,scale=0.01)
        """np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.hidden_dim + self.hidden_dim)),
                                     np.sqrt(6.0 / (self.hidden_dim + self.hidden_dim)),
                                     (self.hidden_dim, self.hidden_dim))
            , dtype=theano.config.floatX)
        """
        W_forget_2 = orthogonal_weight(self.hidden_dim, self.hidden_dim,scale=0.01)
        """np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.hidden_dim + self.hidden_dim)),
                                     np.sqrt(6.0 / (self.hidden_dim + self.hidden_dim)),
                                     (self.hidden_dim, self.hidden_dim))
            , dtype=theano.config.floatX)
        """
        W_output_2 = orthogonal_weight(self.hidden_dim, self.hidden_dim,scale=0.01)
        """
            np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.hidden_dim + self.hidden_dim)),
                                     np.sqrt(6.0 / (self.hidden_dim + self.hidden_dim)),
                                     (self.hidden_dim, self.hidden_dim))
            , dtype=theano.config.floatX)
        """
        U_2 = orthogonal_weight(self.hidden_dim, self.hidden_dim,scale=0.01)
        """np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.hidden_dim + self.hidden_dim)),
                                     np.sqrt(6.0 / (self.hidden_dim + self.hidden_dim)),
                                     (self.hidden_dim, self.hidden_dim))
            , dtype=theano.config.floatX)
        """
        W_2 = orthogonal_weight(self.hidden_dim, self.hidden_dim,scale=0.01)
        """np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.hidden_dim + self.hidden_dim)),
                                     np.sqrt(6.0 / (self.hidden_dim + self.hidden_dim)),
                                     (self.hidden_dim, self.hidden_dim))
            , dtype=theano.config.floatX)
        """


        self.W_2 = theano.shared(value=W_2, name="W_2" , borrow="True")
        self.U_2 = theano.shared(value=U_2, name="U_2" , borrow="True")

        self.W_input_2 = theano.shared(value=W_input_2, name="W_input_2" , borrow="True")
        self.U_input_2 = theano.shared(value=U_input_2, name="U_input_2" , borrow="True")

        self.W_output_2 = theano.shared(value=W_output_2, name="W_output_2" , borrow="True")
        self.U_output_2 = theano.shared(value=U_output_2, name="U_output_2" , borrow="True")

        self.W_forget_2 = theano.shared(value=W_forget_2, name="W_forget_2" , borrow="True")
        self.U_forget_2 = theano.shared(value=U_forget_2, name="U_forget_2" , borrow="True")





        O_w = orthogonal_weight(self.output_dim, self.hidden_dim)
        """np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.output_dim + self.hidden_dim)),
                                     np.sqrt(6.0 / (self.output_dim + self.hidden_dim)),
                                     (self.output_dim, self.hidden_dim))
            , dtype=theano.config.floatX)
        """
        self.O_w = theano.shared(value=O_w, name="O_w" , borrow="True")

        self.params = [self.U_input, self.U_forget, self.U_output, self.W_input, self.W_forget, self.W_output,
                       self.U, self.W,
                       self.U_input_2, self.U_forget_2, self.U_output_2, self.W_input_2, self.W_forget_2, self.W_output_2,
                       self.U_2, self.W_2,
                       self.O_w
                       ]


    def define_network(self):

        Y = T.matrix('output')
        X = T.matrix('input')
        H = T.vector('init_state')

        self.init_lstm_weights()

        """
            def forward_step(output,prev_state, prev_content,prev_state_2,prev_content_2):
            input_gate = T.nnet.hard_sigmoid(
                T.dot((self.U_input), output) + T.dot(self.W_input, prev_state))
            forget_gate = T.nnet.hard_sigmoid(
                T.dot((self.U_forget), output) + T.dot(self.W_forget, prev_state))
            output_gate = T.nnet.hard_sigmoid(
                T.dot((self.U_output), output) + T.dot(self.W_output, prev_state))

            stabilized_input = T.tanh(T.dot((self.U), output) + T.dot(self.W, prev_state))
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







        [self.output,self.hidden_state,self.memory_content, _, _,self.input_gate, self.forget_gate,
         self.output_gate] , updates = theano.scan(
            forward_step,
            sequences=[],
            truncate_gradient=-1,
            outputs_info= [dict(initial=self.EmptyLetter),H,
                               dict(initial=T.zeros(self.hidden_dim,dtype=theano.config.floatX)),
                               dict(initial=T.zeros(self.hidden_dim,dtype=theano.config.floatX)),
                               dict(initial=T.zeros(self.hidden_dim,dtype=theano.config.floatX))
                               , None, None, None
                               ],
            n_steps=3)

        """

        def forward_step(x_t, prev_state, prev_content, prev_state_2, prev_content_2):
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

            return [o, s, c, s2, c2, input_gate, forget_gate, output_gate], theano.scan_module.until(T.eq(T.argmax(o),27))

        [self.output, self.hidden_state, self.memory_content, _, _, self.input_gate, self.forget_gate,
         self.output_gate], updates = theano.scan(
            forward_step,
            sequences=[X],
            truncate_gradient=-1,
            outputs_info=[None,dict(initial=T.zeros(self.hidden_dim, dtype=theano.config.floatX)),
                          dict(initial=T.zeros(self.hidden_dim, dtype=theano.config.floatX)),
                          dict(initial=T.zeros(self.hidden_dim, dtype=theano.config.floatX)),
                          dict(initial=T.zeros(self.hidden_dim, dtype=theano.config.floatX))
                , None, None, None
                          ])


        output = self.output#T.nnet.softmax(T.dot(self.W_out,T.sum(self.output.T,axis=1)))[0]
        self.predict = theano.function([X],[output])

        params = self.params  # + self.output_params
        cost = 1 / T.sum(T.nnet.categorical_crossentropy(T.clip(output, 0.001, 0.999),Y))  #
        grads = T.grad(cost, params)
        updates = [(param_i, param_i - self.learning_rate * cost * param_i + np.random.rand()) for param_i, grad_i in zip(params, grads)]
        self.backprop_update = theano.function([X,Y], cost, updates=updates)
        cost2 = T.sum(T.nnet.categorical_crossentropy(output,Y))
        grads2 = T.grad(cost2,params)
        feedback_updates = [(param_i, param_i - self.learning_rate * grad_i) for param_i, grad_i in zip(params, grads2)]
        self.backprop_update_with_feedback = theano.function([X, Y], [cost2], updates=feedback_updates)






def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


Alpha = ['#','A', 'B', 'C', 'D', 'E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','$']


def get_string(list_of_vec):
    str = " "
    for i in np.arange(len(list_of_vec)):
        str += Alpha[np.argmax(list_of_vec[i])]

    return str


if __name__ == '__main__':
    cp = CaptionGenerator(input_dim=4096,hidden_dim=1000,output_dim=27)
    cp.define_network()

    images, thumb_images = load_images_from_folder("shapes")

    for i in np.arange(10):
        image_embedding = cp.image_reader.get_representation(images[i])[0]
        j = np.random.randint(len(images))
        while j == i:
            j = np.random.randint(len(images))
        random_image_embedding = cp.image_reader.get_representation(images[j])[0]
        input = np.repeat([image_embedding],3,axis=0)
        print(input.shape)
        sequence = cp.predict(input)
        input2 = np.repeat([random_image_embedding], 1, axis=0)
        random_sequence = cp.predict(input2)
        print(str(i)+" "+str(j))
        print(get_string( np.ndarray.tolist(sequence[0])))
        #cost = cp.backprop_update([image_embedding],random_sequence[0])
        #print(cost)