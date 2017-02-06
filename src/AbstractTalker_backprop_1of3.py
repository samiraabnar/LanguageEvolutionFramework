import theano
import theano.tensor as T
import numpy as np
from scipy.spatial import distance


import sys

sys.path.append('../../')

from theanolm.network.weightfunctions import *
from lasagne.updates import sgd, apply_momentum

from generate_data import *

class TalkerLSTM(object):
    def __init__(self,
                 image_rep_dim,
                 lstm_input_dim,
                 lstm_hidden_dim,
                 output_dim,
                 learning_rate,
                 dropout_rate,
                 input_dropout_rate,
                 end_of_sentence_label_max):
        self.image_rep_dim = image_rep_dim
        self.lstm_input_dim = lstm_input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.init_lstm_weights()
        self.dropout_rate = dropout_rate
        self.input_dropout_rate = input_dropout_rate
        self.random_state = np.random.RandomState(23455)

        self.end_of_sentence_label_max = end_of_sentence_label_max

        #self.image_reader = VGG_16('vgg16_weights.h5')
        #self.image_reader.model.compile(optimizer='adam', loss='categorical_crossentropy')


    def init_lstm_weights(self):
        WordEmbedding = random_normal_matrix((self.lstm_input_dim, self.output_dim))
        ImageEmbedding = random_normal_matrix((self.lstm_input_dim, self.image_rep_dim))

        self.WordEmbedding = theano.shared(value=WordEmbedding, name="WordEmbedding", borrow="True")
        self.ImageEmbedding = theano.shared(value=ImageEmbedding, name="ImageEmbedding", borrow="True")

        U_input = random_normal_matrix((self.lstm_hidden_dim, self.lstm_input_dim))

        U_forget = random_normal_matrix((self.lstm_hidden_dim, self.lstm_input_dim))

        U_output = random_normal_matrix((self.lstm_hidden_dim, self.lstm_input_dim))

        W_input = random_normal_matrix((self.lstm_hidden_dim, self.lstm_hidden_dim))

        W_forget = random_normal_matrix((self.lstm_hidden_dim, self.lstm_hidden_dim))

        W_output = random_normal_matrix((self.lstm_hidden_dim, self.lstm_hidden_dim))

        U = random_normal_matrix((self.lstm_hidden_dim, self.lstm_input_dim))

        W = random_normal_matrix((self.lstm_hidden_dim, self.lstm_hidden_dim))

        self.W = theano.shared(value=W, name="W" , borrow="True")
        self.U = theano.shared(value=U, name="U" , borrow="True")

        self.W_input = theano.shared(value=W_input, name="W_input" , borrow="True")
        self.U_input = theano.shared(value=U_input, name="U_input" , borrow="True")

        self.W_output = theano.shared(value=W_output, name="W_output" , borrow="True")
        self.U_output = theano.shared(value=U_output, name="U_output" , borrow="True")

        self.W_forget = theano.shared(value=W_forget, name="W_forget" , borrow="True")
        self.U_forget = theano.shared(value=U_forget, name="U_forget" , borrow="True")


        U_input_2 = random_normal_matrix((self.lstm_hidden_dim, self.lstm_hidden_dim))

        U_forget_2 = random_normal_matrix((self.lstm_hidden_dim, self.lstm_hidden_dim))

        U_output_2 = random_normal_matrix((self.lstm_hidden_dim, self.lstm_hidden_dim))

        W_input_2 = random_normal_matrix((self.lstm_hidden_dim, self.lstm_hidden_dim))

        W_forget_2 = random_normal_matrix((self.lstm_hidden_dim, self.lstm_hidden_dim))

        W_output_2 = random_normal_matrix((self.lstm_hidden_dim, self.lstm_hidden_dim))

        U_2 = random_normal_matrix((self.lstm_hidden_dim, self.lstm_hidden_dim))

        W_2 = random_normal_matrix((self.lstm_hidden_dim, self.lstm_hidden_dim))



        self.W_2 = theano.shared(value=W_2, name="W_2" , borrow="True")
        self.U_2 = theano.shared(value=U_2, name="U_2" , borrow="True")

        self.W_input_2 = theano.shared(value=W_input_2, name="W_input_2" , borrow="True")
        self.U_input_2 = theano.shared(value=U_input_2, name="U_input_2" , borrow="True")

        self.W_output_2 = theano.shared(value=W_output_2, name="W_output_2" , borrow="True")
        self.U_output_2 = theano.shared(value=U_output_2, name="U_output_2" , borrow="True")

        self.W_forget_2 = theano.shared(value=W_forget_2, name="W_forget_2" , borrow="True")
        self.U_forget_2 = theano.shared(value=U_forget_2, name="U_forget_2" , borrow="True")





        O_w = random_normal_matrix((self.output_dim, self.lstm_hidden_dim))
        self.O_w = theano.shared(value=O_w, name="O_w" , borrow="True")

        self.params = [self.U_input, self.U_forget, self.U_output, self.W_input, self.W_forget, self.W_output,
                       self.U, self.W,
                       self.U_input_2, self.U_forget_2, self.U_output_2, self.W_input_2, self.W_forget_2, self.W_output_2,
                       self.U_2, self.W_2,
                       self.O_w, self.WordEmbedding, self.ImageEmbedding
                       ]


    def define_network(self):

        Y = T.matrix('output')
        H = T.vector('init_state')

        self.init_lstm_weights()

        def D(x):
            if self.dropout_rate == 0:
                return x
            else:
                retain_prob = 1 - self.dropout_rate
                return x * np.random.binomial(1, retain_prob, self.lstm_hidden_dim).astype(dtype=np.float32)

        def Input_D(x):
            if self.input_dropout_rate == 0:
                return x
            else:
                retain_prob = 1 - self.dropout_rate
                return x * np.random.binomial(1, retain_prob, self.lstm_input_dim).astype(dtype=np.float32)

        def forward_step(x_t, prev_state, prev_content, prev_state_2, prev_content_2):
            input_gate = T.nnet.hard_sigmoid(
                T.dot(Input_D(self.U_input), x_t) + T.dot(D(self.W_input), prev_state))
            forget_gate = T.nnet.hard_sigmoid(
                T.dot(Input_D(self.U_forget), x_t) + T.dot(D(self.W_forget), prev_state))
            output_gate = T.nnet.hard_sigmoid(
                T.dot(Input_D(self.U_output), x_t) + T.dot(D(self.W_output), prev_state))

            stabilized_input = T.tanh(T.dot(Input_D(self.U), x_t) + T.dot(D(self.W), prev_state))
            c = forget_gate * prev_content + input_gate * stabilized_input
            s = output_gate * T.tanh(c)

            input_gate2 = T.nnet.hard_sigmoid(
                T.dot(D(self.U_input_2), s) + T.dot(D(self.W_input_2), prev_state_2))
            forget_gate2 = T.nnet.hard_sigmoid(
                T.dot(D(self.U_forget_2), s) + T.dot(D(self.W_forget_2), prev_state_2))
            output_gate2 = T.nnet.hard_sigmoid(
                T.dot(D(self.U_output_2), s) + T.dot(D(self.W_output_2), prev_state_2))

            stabilized_input2 = T.tanh(T.dot(D(self.U_2), s) + T.dot(D(self.W_2), prev_state_2))
            c2 = forget_gate2 * prev_content_2 + input_gate2 * stabilized_input2
            s2 = output_gate2 * T.tanh(c2)

            i = T.nnet.softmax(T.dot(self.O_w, s2))[0]
            o = T.dot(self.WordEmbedding,i)

            return [i,o, s, c, s2, c2, input_gate, forget_gate, output_gate], theano.scan_module.until(T.eq(np.argmax(i),self.end_of_sentence_label_max))

        [self.output, _,self.hidden_state, self.memory_content, _, _, self.input_gate, self.forget_gate,
         self.output_gate], updates = theano.scan(
            forward_step,
            #sequences=[X],
            truncate_gradient=-1,
            n_steps= 5,
            outputs_info=[None,dict(initial = T.dot(self.ImageEmbedding,H)),dict(initial= T.zeros(self.lstm_hidden_dim, dtype=theano.config.floatX)),
                          dict(initial=T.zeros(self.lstm_hidden_dim, dtype=theano.config.floatX)),
                          dict(initial= T.zeros(self.lstm_hidden_dim, dtype=theano.config.floatX)),
                          dict(initial=T.zeros(self.lstm_hidden_dim, dtype=theano.config.floatX))
                , None, None, None
                          ])

        self.predict = theano.function([H], [self.output])

        params = self.params

        length = T.max([Y.shape[0], self.output.shape[0]])

        padded_output = T.zeros((length, Y.shape[1]))
        padded_output = T.set_subtensor(padded_output[0:self.output.shape[0], :], self.output)
        padded_output = T.set_subtensor(padded_output[self.output.shape[0]:, :],
                                        T.zeros(self.output_dim, "float32"))

        padded_Y = T.zeros((length, Y.shape[1]))
        padded_Y = T.set_subtensor(padded_Y[0:Y.shape[0], :], Y)
        padded_Y = T.set_subtensor(padded_Y[Y.shape[0]:, :], T.zeros(self.output_dim,"float32"))

        lambda_1 = pow(10, -(T.log((self.lstm_hidden_dim + self.lstm_input_dim) / 2) / T.log(10)) - 2)
        lambda_2 = pow(10, -(T.log(self.lstm_hidden_dim) / T.log(10)) * 2 - 2)

        cost = T.sum(T.nnet.categorical_crossentropy(padded_output + 0.00000001, padded_Y)) + lambda_1 * sum(
            [T.sum(param ** 2) for param in params]) + lambda_2 * sum([T.sum(abs(param) > 0) for param in params])
        updates_sgd = sgd(cost, params, learning_rate=self.learning_rate)
        updates = apply_momentum(updates_sgd, params, momentum=0.9)

        self.backprop_update = theano.function([H, Y], [self.output,cost], updates=updates)
        self.get_output_without_update = theano.function([H, Y], [self.output, cost])

        #backprop_update_with_feedback_negative = theano.function([H, Y], [cost3], updates=feedback_updates_neg)

        """re_inforce_cost = -T.sum(self.output)
        re_inforce_grads = T.grad(re_inforce_cost,params)
        p_updates = [(param_i, param_i + self.learning_rate * grad_i +  grad_i * np.random.rand()) for grad_i,param_i in
                     zip(re_inforce_grads, params)]
        n_updates = [(param_i, param_i - self.learning_rate * grad_i +  grad_i * np.random.rand()) for grad_i,param_i in
                     zip(re_inforce_grads,params)]

        self.positive_reinforce_update = theano.function([H],[],updates=p_updates)
        self.negetive_reinforce_update = theano.function([H],[],updates=n_updates)
        """







def get_string(list_of_vec,VOCAB,end_index):
    str = " "
    for i in np.arange(len(list_of_vec)):
        if np.argmax(list_of_vec[i]) != end_index:
            a = list(list_of_vec[i])
            a[end_index] = 0
            a = a /sum(a)
            str += " "+VOCAB[np.random.choice(len(a), 1,p=a)[0]]
        else:
            str += " " + VOCAB[np.argmax(list_of_vec[i])]
    return str.strip()

def get_probabilistic_sequence(list_of_vec,end_index):
    seq = []
    for i in np.arange(len(list_of_vec)):
        if np.argmax(list_of_vec[i]) != end_index:
            a = list(list_of_vec[i])
            a[end_index] = 0
            a = a / sum(a)
            seq.append(np.eye(len(list_of_vec[i]))[np.random.choice(len(a), 1, p=a)[0]])
        else:
            seq.append(np.eye(len(list_of_vec[i]))[np.argmax(list_of_vec[i])])
    return seq



if __name__ == '__main__':

    VOCAB = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', '<\s>']
    Dic = {}
    for i in np.arange(len(VOCAB)):
        Dic[VOCAB[i]] = i
    ONE_HOT_VECS = np.eye(len(VOCAB))

    number_of_concepts = 2
    number_of_values_per_concept = 2
    number_of_items_per_combination = 2
    label_dim = number_of_concepts * number_of_values_per_concept + 1
    dg = DataSetGenerator(number_of_concepts, number_of_values_per_concept)
    dg.generate_data_with_marginal_labels(number_of_items_per_label=number_of_items_per_combination, core_item=0)
    items = dg.combined_items
    labels = dg.combined_labels

    talker = TalkerLSTM(image_rep_dim=number_of_concepts * number_of_values_per_concept * number_of_items_per_combination,
                        lstm_input_dim=100,  # len(VOCAB),
                               lstm_hidden_dim=128,
                        output_dim=label_dim,
                        learning_rate=0.0005,
                        dropout_rate=0.5,
                        input_dropout_rate=0.0,
                        end_of_sentence_label_max=np.argmax(labels[0][-1]))

    talker.define_network()



    print("Network defined :)")

    import time
    import random

    start = time.time()
    number_of_epochs = 1000
    for e in np.arange(number_of_epochs):
        for k in np.arange(len(items)):
            output,cost = talker.backprop_update(np.asarray(items[k],dtype="float32"),np.asarray(labels[k],dtype="float32"))
            print(get_string(labels[k],VOCAB,np.argmax(labels[0][-1]))+" : "+get_string(output,VOCAB,np.argmax(labels[0][-1])))
            print(cost)




    end = time.time()
    print ("total time: "+str(end - start))