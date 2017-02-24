import theano
import theano.tensor as T
import numpy as np
from scipy.spatial import distance
import sklearn
from theanolm.network import weightfunctions
from lasagne.updates import sgd, apply_momentum,adam
from generate_data import *


#from AbstractTalker_backprop_1of3 import *


class ListernerLSTM(object):
    def __init__(self, input_dim, hidden_dim, output_dim,dropout_rate,input_dropout_rate):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = 0.01
        self.dropout_rate = dropout_rate
        self.input_dropout_rate = input_dropout_rate
        self.random_state = np.random.RandomState(23455)
        self.initial_hiddens = [None,dict(initial=T.zeros(self.hidden_dim, dtype=theano.config.floatX)),
                                     dict(initial=T.zeros(self.hidden_dim, dtype=theano.config.floatX))
                          , None, None, None
                          ]


    def init_lstm_weights(self):
        U_input = weightfunctions.random_normal_matrix((self.hidden_dim, self.input_dim))

        U_forget = weightfunctions.random_normal_matrix((self.hidden_dim, self.input_dim))

        U_output = weightfunctions.random_normal_matrix((self.hidden_dim, self.input_dim))

        W_input = weightfunctions.random_normal_matrix((self.hidden_dim, self.hidden_dim))

        W_forget = weightfunctions.random_normal_matrix((self.hidden_dim, self.hidden_dim))

        W_output = weightfunctions.random_normal_matrix((self.hidden_dim, self.hidden_dim))

        U = weightfunctions.random_normal_matrix((self.hidden_dim, self.input_dim))

        W = weightfunctions.random_normal_matrix((self.hidden_dim, self.hidden_dim))



        self.W = theano.shared(value=W, name="W" , borrow="True")
        self.U = theano.shared(value=U, name="U" , borrow="True")

        self.W_input = theano.shared(value=W_input, name="W_input" , borrow="True")
        self.U_input = theano.shared(value=U_input, name="U_input" , borrow="True")

        self.W_output = theano.shared(value=W_output, name="W_output" , borrow="True")
        self.U_output = theano.shared(value=U_output, name="U_output" , borrow="True")

        self.W_forget = theano.shared(value=W_forget, name="W_forget" , borrow="True")
        self.U_forget = theano.shared(value=U_forget, name="U_forget" , borrow="True")


        U_input_2 = weightfunctions.random_normal_matrix((self.hidden_dim, self.hidden_dim))

        U_forget_2 = weightfunctions.random_normal_matrix((self.hidden_dim, self.hidden_dim))

        U_output_2 = weightfunctions.random_normal_matrix((self.hidden_dim, self.hidden_dim))

        W_input_2 = weightfunctions.random_normal_matrix((self.hidden_dim, self.hidden_dim))

        W_forget_2 = weightfunctions.random_normal_matrix((self.hidden_dim, self.hidden_dim))

        W_output_2 = weightfunctions.random_normal_matrix((self.hidden_dim, self.hidden_dim))

        U_2 = weightfunctions.random_normal_matrix((self.hidden_dim, self.hidden_dim))

        W_2 = weightfunctions.random_normal_matrix((self.hidden_dim, self.hidden_dim))


        self.W_2 = theano.shared(value=W_2, name="W_2" , borrow="True")
        self.U_2 = theano.shared(value=U_2, name="U_2" , borrow="True")

        self.W_input_2 = theano.shared(value=W_input_2, name="W_input_2" , borrow="True")
        self.U_input_2 = theano.shared(value=U_input_2, name="U_input_2" , borrow="True")

        self.W_output_2 = theano.shared(value=W_output_2, name="W_output_2" , borrow="True")
        self.U_output_2 = theano.shared(value=U_output_2, name="U_output_2" , borrow="True")

        self.W_forget_2 = theano.shared(value=W_forget_2, name="W_forget_2" , borrow="True")
        self.U_forget_2 = theano.shared(value=U_forget_2, name="U_forget_2" , borrow="True")





        O_w = weightfunctions.random_normal_matrix((self.output_dim, self.hidden_dim))

        self.O_w = theano.shared(value=O_w, name="O_w" , borrow="True")

        self.params = [self.U_input, self.U_forget, self.U_output, self.W_input, self.W_forget, self.W_output,
                       self.U, self.W,
                       self.U_input_2, self.U_forget_2, self.U_output_2, self.W_input_2, self.W_forget_2, self.W_output_2,
                       self.U_2, self.W_2
                       ,self.O_w
                       ]




    def define_network(self):

        X = T.matrix('input')
        Y = T.vector('output')
        all_targets = T.matrix("targets")

        #H = T.vector('init_state')

        self.init_lstm_weights()

        def D(x):
            if self.dropout_rate == 0:
                return x
            else:
                retain_prob = 1 - self.dropout_rate
                return x * np.random.binomial(1, retain_prob, self.hidden_dim).astype(dtype=np.float32)

        def Input_D(x):
            if self.input_dropout_rate == 0:
                return x
            else:
                retain_prob = 1 - self.dropout_rate
                return x * np.random.binomial(1, retain_prob, self.input_dim).astype(dtype=np.float32)

        def forward_step(x_t, prev_state, prev_content,prev_state_2,prev_content_2):
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



            o = T.dot(self.O_w, s2)

            return [o, s, c, s2, c2, input_gate, forget_gate, output_gate]




        [self.output,self.hidden_state,self.memory_content, _, _,self.input_gate, self.forget_gate,
         self.output_gate] , updates = theano.scan(
            forward_step,
            sequences=[X],
            truncate_gradient=-1,
            outputs_info= [None,dict(initial=T.zeros(self.hidden_dim, dtype=theano.config.floatX)),
                               dict(initial=T.zeros(self.hidden_dim, dtype=theano.config.floatX)),
                               dict(initial=T.zeros(self.hidden_dim, dtype=theano.config.floatX)),
                               dict(initial=T.zeros(self.hidden_dim, dtype=theano.config.floatX))
                               , None, None, None
                               ])




        output = T.nnet.sigmoid (self.output[-1]) #T.nnet.softmax(T.dot(self.W_out,T.sum(self.output.T,axis=1)))[0]

        self.predict = theano.function([X],[output])

        params = self.params #+ self.output_params
        lambda_L1 = 0.001
        cost =  T.sum(T.nnet.binary_crossentropy(T.clip(output, 1e-7, 1.0 - 1e-7),Y)) + lambda_L1 * T.sum(([ T.abs(p) for p in params]))


        updates =  adam(cost,params,learning_rate=0.0001) #apply_momentum(updates_sgd, params, momentum=0.9)
        self.backprop_update = theano.function([X,Y],[output,cost],updates=updates)





if __name__ == '__main__':
    VOCAB = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', '<\s>']
    Dic = {}
    for i in np.arange(len(VOCAB)):
        Dic[VOCAB[i]] = i
    ONE_HOT_VECS = np.eye(len(VOCAB))

    number_of_concepts = 2
    number_of_values_per_concept = 3
    number_of_items_per_combination = 2
    label_dim = number_of_concepts * number_of_values_per_concept + 1
    dg = DataSetGenerator(number_of_concepts, number_of_values_per_concept)
    #dg.generate_data_with_marginal_labels(number_of_items_per_label=number_of_items_per_combination, core_item=0)
    dg.generate_data()
    items = dg.items
    labels = dg.labels

    total_count = len(items)
    test_count = (total_count // 5)

    train_items = items[:-test_count]
    test_items = items[-test_count:]
    train_labels = labels[:-test_count]
    test_labels = labels[-test_count:]

    lstm_listener = ListernerLSTM(input_dim = label_dim,
                                  hidden_dim = 128,
                                  output_dim = number_of_values_per_concept*number_of_concepts,
                                  dropout_rate=0.0,
                                  input_dropout_rate=0
                                  )
    lstm_listener.define_network()

    import time
    import random

    start = time.time()
    number_of_epochs = 3000
    for e in np.arange(number_of_epochs):
        train_items_index = np.arange(len(train_labels))
        np.random.shuffle(train_items_index)
        train_cost = []
        for k in train_items_index:
            all_other_index= list(train_items_index[:])
            all_other_index.remove(k)
            all_other_index = np.asarray(all_other_index)

            all_other_items = np.asarray(train_items,"float32")[all_other_index,:]
            """item = []
            shuffled_index = []
            for s in np.arange(number_of_items_per_combination):
                item.append(items[k][s*(number_of_values_per_concept*number_of_concepts):(s+1)*(number_of_values_per_concept*number_of_concepts)])
                shuffled_index.append(s)

            np.random.shuffle(shuffled_index)
            shuffled_item = []
            for i in shuffled_index:
                shuffled_item.append(item[i])

            target_label = np.eye(number_of_items_per_combination,dtype="float32")[shuffled_index.index(0)]
            """

            input_items = train_labels[k] #np.concatenate(tuple(shuffled_item),axis=0)
            #input_items = [np.concatenate((input_items,word),axis=0) for word in labels[k]]

            target_label = train_items[k]

            output, cost = lstm_listener.backprop_update(np.asarray(input_items,dtype="float32")
                                                         , np.asarray(target_label,dtype="float32")
                                                         #, all_other_items
                                                         )

            train_cost.append(cost)

            print(target_label)
            print(output)

            #print(str(e*len(items)+k)+": "+str(shuffled_index.index(0) == np.argmax(output)))

        print("train_cost: " + str(np.mean(train_cost)))
        test_items_index =  np.arange(len(test_items))
        costs = []
        for k in test_items_index:
            input_items = test_labels[k]  # np.concatenate(tuple(shuffled_item),axis=0)
            # input_items = [np.concatenate((input_items,word),axis=0) for word in labels[k]]

            target_label = test_items[k]

            output = lstm_listener.predict(np.asarray(input_items, dtype="float32")
                                                         #np.asarray(target_label, dtype="float32"),
                                                )

            #costs.append(cost)
            print(target_label)
            print(output)
            print("********************")

        #print("test_cost: "+str(np.mean(cost)))


