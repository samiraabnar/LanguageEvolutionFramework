import theano
import theano.tensor as T
import numpy as np
from theanolm.network import weightfunctions
from lasagne.updates import adam
from generate_data import *

from Util import *
from scipy.spatial import *


class ImageGenerator(object):
    def __init__(self, input_dim, hidden_dim, output_dim,dropout_rate,input_dropout_rate,all_items):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = 0.001
        self.dropout_rate = dropout_rate
        self.input_dropout_rate = input_dropout_rate
        self.random_state = np.random.RandomState(23455)
        self.initial_hiddens = [None,dict(initial=T.zeros(self.hidden_dim, dtype=theano.config.floatX)),
                                     dict(initial=T.zeros(self.hidden_dim, dtype=theano.config.floatX))
                          , None, None, None
                          ]

        self.item_tree = cKDTree(all_items)
        self.all_items = all_items
        self.test = False

    def set_allitems(self,all_items):
        self.item_tree = cKDTree(all_items)
        self.all_items = all_items

    def init_lstm_weights(self):
        U_input = weightfunctions.random_normal_matrix((self.hidden_dim, self.input_dim),scale=1.0)

        U_forget = weightfunctions.random_normal_matrix((self.hidden_dim, self.input_dim),scale=1.0)

        U_output = weightfunctions.random_normal_matrix((self.hidden_dim, self.input_dim),scale=1.0)

        W_input = weightfunctions.random_normal_matrix((self.hidden_dim, self.hidden_dim),scale=1.0)

        W_forget = weightfunctions.random_normal_matrix((self.hidden_dim, self.hidden_dim),scale=1.0)

        W_output = weightfunctions.random_normal_matrix((self.hidden_dim, self.hidden_dim),scale=1.0)

        U = weightfunctions.random_normal_matrix((self.hidden_dim, self.input_dim),scale=1.0)

        W = weightfunctions.random_normal_matrix((self.hidden_dim, self.hidden_dim),scale=1.0)



        self.W = theano.shared(value=W, name="W" , borrow="True")
        self.U = theano.shared(value=U, name="U" , borrow="True")

        self.W_input = theano.shared(value=W_input, name="W_input" , borrow="True")
        self.U_input = theano.shared(value=U_input, name="U_input" , borrow="True")

        self.W_output = theano.shared(value=W_output, name="W_output" , borrow="True")
        self.U_output = theano.shared(value=U_output, name="U_output" , borrow="True")

        self.W_forget = theano.shared(value=W_forget, name="W_forget" , borrow="True")
        self.U_forget = theano.shared(value=U_forget, name="U_forget" , borrow="True")


        U_input_2 = weightfunctions.random_normal_matrix((self.hidden_dim, self.hidden_dim),scale=1.0)

        U_forget_2 = weightfunctions.random_normal_matrix((self.hidden_dim, self.hidden_dim),scale=1.0)

        U_output_2 = weightfunctions.random_normal_matrix((self.hidden_dim, self.hidden_dim),scale=1.0)

        W_input_2 = weightfunctions.random_normal_matrix((self.hidden_dim, self.hidden_dim),scale=1.0)

        W_forget_2 = weightfunctions.random_normal_matrix((self.hidden_dim, self.hidden_dim),scale=1.0)

        W_output_2 = weightfunctions.random_normal_matrix((self.hidden_dim, self.hidden_dim),scale=1.0)

        U_2 = weightfunctions.random_normal_matrix((self.hidden_dim, self.hidden_dim),scale=1.0)

        W_2 = weightfunctions.random_normal_matrix((self.hidden_dim, self.hidden_dim),scale=1.0)


        self.W_2 = theano.shared(value=W_2, name="W_2" , borrow="True")
        self.U_2 = theano.shared(value=U_2, name="U_2" , borrow="True")

        self.W_input_2 = theano.shared(value=W_input_2, name="W_input_2" , borrow="True")
        self.U_input_2 = theano.shared(value=U_input_2, name="U_input_2" , borrow="True")

        self.W_output_2 = theano.shared(value=W_output_2, name="W_output_2" , borrow="True")
        self.U_output_2 = theano.shared(value=U_output_2, name="U_output_2" , borrow="True")

        self.W_forget_2 = theano.shared(value=W_forget_2, name="W_forget_2" , borrow="True")
        self.U_forget_2 = theano.shared(value=U_forget_2, name="U_forget_2" , borrow="True")





        O_w = weightfunctions.random_normal_matrix((self.output_dim, self.hidden_dim),scale=1.0)

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
            if self.dropout_rate == 0 or self.test == True:
                return x
            else:
                retain_prob = 1 - self.dropout_rate
                return x * np.random.binomial(1, retain_prob, self.hidden_dim).astype(dtype=np.float32)

        def Input_D(x):
            if self.input_dropout_rate == 0 or self.test == True:
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



        params = self.params #+ self.output_params
        lambda_L1 = 0.0001
        L1_Loss = lambda_L1 * T.sum([ T.sum(abs(p)) for p in params])
        cost =  T.sum(T.nnet.binary_crossentropy(T.clip(output, 1e-7, 1.0 - 1e-7),Y)) + L1_Loss


        updates =  adam(cost,params,learning_rate=self.learning_rate) #apply_momentum(updates_sgd, params, momentum=0.9)
        self.backprop_update = theano.function([X,Y],[output,cost],updates=updates)

        self.predict = theano.function([X,Y], [output, cost])
        self.generate_item = theano.function([X], [output])
        self.get_cost = theano.function([X,Y], [cost])


    def retrieve_image(self,label):
        [reconstructed_item] = self.generate_item(label)
        dd, ii = self.item_tree.query(reconstructed_item)

        return np.asarray(self.all_items[ii],dtype="float32"), ii

    def calculate_accuracy(self,predictions,targets):
        corrects = 0.0
        for p,t in zip(predictions,targets):
            dd, ii = self.item_tree.query(p)
            if (self.all_items[ii] == t).all():
                corrects += 1

        return corrects / len(predictions)


class Experiment(object):
    def __init__(self,id,relative_test_size,number_of_concepts,number_of_values_per_concept,number_of_items_per_combination):
        self.id = id

        self.number_of_concepts = number_of_concepts
        self.number_of_values_per_concept = number_of_values_per_concept
        self.number_of_items_per_combination = number_of_items_per_combination

        self.relative_test_size = relative_test_size
        self.test_items = []
        self.train_items = []

        self.iteration_train_cost = []
        self.iteration_test_cost = []
        self.iteration_test_accuracy = []
        self.iteration_train_accuracy = []


    def prepare_data(self):
        number_of_concepts = self.number_of_concepts
        number_of_values_per_concept = self.number_of_values_per_concept
        self.label_dim = number_of_concepts * number_of_values_per_concept + 1

        dg = DataSetGenerator(number_of_concepts, number_of_values_per_concept)
        dg.generate_data()
        self.items = list(dg.items)
        self.labels = list(dg.labels)

        indexes = np.arange(len(self.items))
        np.random.shuffle(indexes)

        self.items = np.asarray(self.items)[indexes, :]
        self.labels = np.asarray(self.labels)[indexes]

        total_count = len(self.items)
        test_count = (total_count // self.relative_test_size)

        self.train_items = self.items[:-test_count]
        self.test_items = self.items[-test_count:]
        self.train_labels = self.labels[:-test_count]
        self.test_labels = self.labels[-test_count:]
        self.last_index = np.argmax(dg.labels[0][-1])

    def save(self):
        pass

    def print_ordered_items_names(self):
        for i in np.arange(len(self.items)):
            print(get_string(self.labels[i], VOCAB, self.last_index))


def do_the_exp(exp):
    lstm_listener = ImageGenerator(input_dim = exp.label_dim,
                                   hidden_dim = 256,
                                   output_dim = exp.number_of_values_per_concept*exp.number_of_concepts,
                                   dropout_rate=0.9,
                                   input_dropout_rate=.0
                                   , all_items=exp.items)

    lstm_listener.define_network()

    number_of_epochs = 500

    for e in np.arange(number_of_epochs):
        train_items_index = np.arange(len(exp.train_labels))
        np.random.shuffle(train_items_index)
        train_costs = []
        train_predictions = []
        train_targets = []
        for k in train_items_index:
            all_other_index= list(train_items_index[:])
            all_other_index.remove(k)
            all_other_index = np.asarray(all_other_index)

            all_other_items = np.asarray(exp.train_items,"float32")[all_other_index,:]
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
            input_items = exp.train_labels[k]
            target_label = exp.train_items[k]
            output, cost = lstm_listener.backprop_update(np.asarray(input_items,dtype="float32")
                                                         , np.asarray(target_label,dtype="float32")
                                                         #, all_other_items
                                                         )

            train_costs.append(cost)
            train_predictions.append(output)
            train_targets.append(np.asarray(target_label,dtype="float32"))

        test_items_index =  np.arange(len(exp.test_items))
        test_costs = []
        predictions = []
        targets = []
        lstm_listener.test = True
        for k in test_items_index:
            input_items = exp.test_labels[k]  # np.concatenate(tuple(shuffled_item),axis=0)
            # input_items = [np.concatenate((input_items,word),axis=0) for word in labels[k]]

            target_label = exp.test_items[k]

            output, cost = lstm_listener.predict(np.asarray(input_items, dtype="float32"),
                                                 np.asarray(target_label, dtype="float32")
                                                )
            predictions.append(output)
            targets.append(np.asarray(target_label, dtype="float32"))

            test_costs.append(cost)
        lstm_listener.test = False
        test_accuracy = lstm_listener.calculate_accuracy(predictions,targets)
        train_accuracy = lstm_listener.calculate_accuracy(train_predictions, train_targets)
        print("train_cost: " + str(np.mean(train_costs)))
        print("test_cost: "+str(np.mean(test_costs)))
        print("train accuracy: " + str(train_accuracy))
        print("test accuracy: "+str(test_accuracy))

        exp.iteration_train_cost.append(np.mean(train_costs))
        exp.iteration_test_cost.append(np.mean(test_costs))
        exp.iteration_test_accuracy.append(test_accuracy)
        exp.iteration_train_accuracy.append(train_accuracy)


if __name__ == '__main__':
    VOCAB = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', '<\s>']
    Dic = {}
    for i in np.arange(len(VOCAB)):
        Dic[VOCAB[i]] = i
    ONE_HOT_VECS = np.eye(len(VOCAB))

    #exp1 exp2 exp3 exp4
    """
    relative_test_sizes = [5,4,3,2]
    for i in np.arange(len(relative_test_sizes)):
        exp = Experiment(id=i+1,
                relative_test_size=relative_test_sizes[i],
                number_of_concepts=3,
                number_of_values_per_concept=3,
                number_of_items_per_combination=3)

        exp.prepare_data()
        do_the_exp((exp))
    """

    """relative_test_sizes = [5, 4, 3, 2]
    for i in np.arange(len(relative_test_sizes)):
        exp = Experiment(id=i + 100,
                relative_test_size=relative_test_sizes[i],
                number_of_concepts=3,
                number_of_values_per_concept=4,
                number_of_items_per_combination=3)

        exp.prepare_data()
        do_the_exp((exp))

    """

    Plotting.plot_performance(exp.iteration_train_cost, exp.iteration_test_cost)
    Plotting.plot_performance(exp.iteration_train_accuracy, exp.iteration_test_accuracy)