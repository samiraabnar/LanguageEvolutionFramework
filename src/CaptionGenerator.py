import theano.tensor as T
import theano
import numpy as np
from scipy.spatial import distance


import sys

sys.path.append('../../')

from LanguageEvolutionFramework.src.Vision.VGG_16 import *
from theanolm.matrixfunctions import orthogonal_weight

from lasagne.updates import sgd, apply_momentum


class CaptionGenerator(object):

    def __init__(self,image_dim,input_dim,hidden_dim,output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.image_dim = image_dim
        self.learning_rate = 0.001
        self.random_state = np.random.RandomState(23455)

        self.EmptyLetter = np.asarray(np.eye(output_dim)[0],dtype='float32')

        self.image_reader = VGG_16('vgg16_weights.h5')
        self.image_reader.model.compile(optimizer='adam', loss='categorical_crossentropy')


    def init_lstm_weights(self):
        WordEmbedding = orthogonal_weight(self.input_dim, self.output_dim,scale=0.01)
        ImageEmbedding = orthogonal_weight(self.input_dim, self.image_dim,scale=0.01)

        self.WordEmbedding = theano.shared(value=WordEmbedding, name="WordEmbedding", borrow="True")
        self.ImageEmbedding = theano.shared(value=ImageEmbedding, name="ImageEmbedding", borrow="True")

        U_input = orthogonal_weight(self.hidden_dim, self.input_dim,scale=0.01)

        U_forget = orthogonal_weight(self.hidden_dim, self.input_dim,scale=0.01)

        U_output = orthogonal_weight(self.hidden_dim, self.input_dim,scale=0.01)

        W_input = orthogonal_weight(self.hidden_dim, self.hidden_dim,scale=0.01)

        W_forget = orthogonal_weight(self.hidden_dim, self.hidden_dim,scale=0.01)

        W_output = orthogonal_weight(self.hidden_dim, self.hidden_dim,scale=0.01)

        U = orthogonal_weight(self.hidden_dim, self.input_dim,scale=0.01)

        W = orthogonal_weight(self.hidden_dim, self.hidden_dim,scale=0.01)

        self.W = theano.shared(value=W, name="W" , borrow="True")
        self.U = theano.shared(value=U, name="U" , borrow="True")

        self.W_input = theano.shared(value=W_input, name="W_input" , borrow="True")
        self.U_input = theano.shared(value=U_input, name="U_input" , borrow="True")

        self.W_output = theano.shared(value=W_output, name="W_output" , borrow="True")
        self.U_output = theano.shared(value=U_output, name="U_output" , borrow="True")

        self.W_forget = theano.shared(value=W_forget, name="W_forget" , borrow="True")
        self.U_forget = theano.shared(value=U_forget, name="U_forget" , borrow="True")


        U_input_2 = orthogonal_weight(self.hidden_dim, self.hidden_dim)

        U_forget_2 = orthogonal_weight(self.hidden_dim, self.hidden_dim,scale=0.01)

        U_output_2 = orthogonal_weight(self.hidden_dim, self.hidden_dim,scale=0.01)

        W_input_2 = orthogonal_weight(self.hidden_dim, self.hidden_dim,scale=0.01)

        W_forget_2 = orthogonal_weight(self.hidden_dim, self.hidden_dim,scale=0.01)

        W_output_2 = orthogonal_weight(self.hidden_dim, self.hidden_dim,scale=0.01)

        U_2 = orthogonal_weight(self.hidden_dim, self.hidden_dim,scale=0.01)

        W_2 = orthogonal_weight(self.hidden_dim, self.hidden_dim,scale=0.01)



        self.W_2 = theano.shared(value=W_2, name="W_2" , borrow="True")
        self.U_2 = theano.shared(value=U_2, name="U_2" , borrow="True")

        self.W_input_2 = theano.shared(value=W_input_2, name="W_input_2" , borrow="True")
        self.U_input_2 = theano.shared(value=U_input_2, name="U_input_2" , borrow="True")

        self.W_output_2 = theano.shared(value=W_output_2, name="W_output_2" , borrow="True")
        self.U_output_2 = theano.shared(value=U_output_2, name="U_output_2" , borrow="True")

        self.W_forget_2 = theano.shared(value=W_forget_2, name="W_forget_2" , borrow="True")
        self.U_forget_2 = theano.shared(value=U_forget_2, name="U_forget_2" , borrow="True")





        O_w = orthogonal_weight(self.output_dim, self.hidden_dim)
        self.O_w = theano.shared(value=O_w, name="O_w" , borrow="True")

        self.params = [self.U_input, self.U_forget, self.U_output, self.W_input, self.W_forget, self.W_output,
                       self.U, self.W,
                       self.U_input_2, self.U_forget_2, self.U_output_2, self.W_input_2, self.W_forget_2, self.W_output_2,
                       self.U_2, self.W_2,
                       self.O_w, self.WordEmbedding, self.ImageEmbedding
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

            i = T.nnet.softmax(T.dot(self.O_w, s2))[0]
            o = T.dot(self.WordEmbedding,i)

            return [i,o, s, c, s2, c2, input_gate, forget_gate, output_gate], theano.scan_module.until(np.argmax(i) > (self.output_dim - 2))

        [self.output, _,self.hidden_state, self.memory_content, _, _, self.input_gate, self.forget_gate,
         self.output_gate], updates = theano.scan(
            forward_step,
            #sequences=[X],
            truncate_gradient=-1,
            n_steps= 5,
            outputs_info=[None,dict(initial = T.dot(self.ImageEmbedding,H)),dict(initial= T.zeros(self.hidden_dim, dtype=theano.config.floatX)),
                          dict(initial=T.zeros(self.hidden_dim, dtype=theano.config.floatX)),
                          dict(initial= T.zeros(self.hidden_dim, dtype=theano.config.floatX)),
                          dict(initial=T.zeros(self.hidden_dim, dtype=theano.config.floatX))
                , None, None, None
                          ])


        output = self.output#T.nnet.softmax(T.dot(self.W_out,T.sum(self.output.T,axis=1)))[0]
        self.predict = theano.function([H],[output])

        params = self.params  # + self.output_params
        cost = 1 / T.sum(T.nnet.binary_crossentropy(T.clip(output, 0.001, 0.999),Y))  #
        grads = T.grad(cost, params)
        updates = [(param_i, param_i - self.learning_rate * cost * param_i + np.random.rand()) for param_i, grad_i in zip(params, grads)]
        self.backprop_update = theano.function([H,Y], cost, updates=updates)

        padded_output = T.zeros(Y.shape) #+ [np.eye(self.output_dim,dtype="float32")[self.output_dim - 1] for i in np.arange(output.shape[0],Y.shape[0])]
        padded_output = T.set_subtensor(padded_output[0:output.shape[0],:],output)
        padded_output = T.set_subtensor(padded_output[output.shape[0]:, :], T.eye(self.output_dim)[self.output_dim - 1])
        cost2 = T.sum(T.nnet.categorical_crossentropy(padded_output + 0.000001,Y))
        grads2 = T.grad(cost2,params)
        #feedback_updates = [(param_i, param_i - self.learning_rate * grad_i) for param_i, grad_i in zip(params, grads2)]
        updates_sgd = sgd(cost2, params, learning_rate=0.0001)
        feedback_updates = apply_momentum(updates_sgd, params, momentum=0.9)

        self.backprop_update_with_feedback = theano.function([H, Y], [cost2], updates=feedback_updates)
        self.ce_error = theano.function([H, Y], cost2)

        backprop_update_with_feedback_negative = theano.function([H, Y], [cost3], updates=feedback_updates_neg)

    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x, y) for x, y in zip(X, Y)])

    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X, Y) / float(num_words)




def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


Alpha = ['#','A', 'B', 'C', 'D', 'E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','$']


def get_string(list_of_vec):
    str = " "
    for i in np.arange(len(list_of_vec)):
        str += Alpha[np.argmax(list_of_vec[i])]

    return str


if __name__ == '__main__':
    cp = CaptionGenerator(image_dim=4096,input_dim=4096,hidden_dim=1000,output_dim=28)
    cp.define_network()

    images, thumb_images = load_images_from_folder("shapes")
    print("Network defined :)")

    import time

    start = time.time()
    for k in np.arange(1000):
        i = np.random.randint(5)
        image_embedding = cp.image_reader.get_representation(images[i])[0]
        j = np.random.randint(len(images))
        while j == i:
            j = np.random.randint(len(images))
        random_image_embedding = cp.image_reader.get_representation(images[j])[0]
        input = np.repeat([image_embedding],3,axis=0)

        sequence = cp.predict(image_embedding)
        input2 = np.repeat([random_image_embedding], 1, axis=0)
        random_sequence = cp.predict(random_image_embedding)
        print(str(i)+" "+str(j))
        print(get_string( np.ndarray.tolist(sequence[0])))

        d = np.zeros((np.max([3,sequence[0].shape[0]]), 28), dtype='float32')
        d[0][1 + i % 25] = 1.0
        d[1][1 + (i + 1) % 25] = 1.0
        d[2][1 + (i + 2) % 25] = 1.0
        for i in np.arange(3,sequence[0].shape[0],1):
            d[i][27] = 1.0

        print(get_string(d))
        cost = cp.backprop_update_with_feedback(image_embedding,d)
        print(cost)

    end = time.time()
    print ("total time: "+str(end - start))