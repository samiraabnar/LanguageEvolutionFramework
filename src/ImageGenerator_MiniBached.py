import theano
import theano.tensor as T
import numpy as np
from theanolm.network import weightfunctions
from lasagne.updates import adam
from scipy.spatial import *
import pickle

from generate_data import *
from Util import *




class ImageGenerator(object):
    def __init__(self, input_dim, hidden_dim, output_dim,dropout_rate,input_dropout_rate,all_items):
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

        self.item_tree = cKDTree(all_items)
        self.all_items = all_items

    def init_lstm_weights(self):
        U_input = weightfunctions.random_normal_matrix((self.input_dim, self.hidden_dim))

        U_forget = weightfunctions.random_normal_matrix((self.input_dim, self.hidden_dim))

        U_output = weightfunctions.random_normal_matrix((self.input_dim, self.hidden_dim))

        W_input = weightfunctions.random_normal_matrix((self.hidden_dim, self.hidden_dim))

        W_forget = weightfunctions.random_normal_matrix((self.hidden_dim, self.hidden_dim))

        W_output = weightfunctions.random_normal_matrix((self.hidden_dim, self.hidden_dim))

        U = weightfunctions.random_normal_matrix((self.input_dim, self.hidden_dim))

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





        O_w = weightfunctions.random_normal_matrix((self.hidden_dim, self.output_dim))

        self.O_w = theano.shared(value=O_w, name="O_w" , borrow="True")

        ItemEmbedding = weightfunctions.random_normal_matrix((self.output_dim, self.hidden_dim))
        self.ItemEmbedding = theano.shared(value=ItemEmbedding, name="ItemEmbedding" , borrow="True")

        self.params = [self.U_input, self.U_forget, self.U_output, self.W_input, self.W_forget, self.W_output,
                       self.U, self.W,
                       self.U_input_2, self.U_forget_2, self.U_output_2, self.W_input_2, self.W_forget_2, self.W_output_2,
                       self.U_2, self.W_2
                       ,self.O_w
                       ,self.ItemEmbedding
                       ]




    def define_network(self):

        X = T.tensor3('input')
        Y = T.matrix('output')
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
                T.dot(x_t,Input_D(self.U_input)) + T.dot(prev_state,D(self.W_input)))
            forget_gate = T.nnet.hard_sigmoid(
                T.dot(x_t,Input_D(self.U_forget)) + T.dot(prev_state,D(self.W_forget)))
            output_gate = T.nnet.hard_sigmoid(
                T.dot(x_t,Input_D(self.U_output)) + T.dot(prev_state,D(self.W_output)))

            stabilized_input = T.tanh(T.dot(x_t,Input_D(self.U)) + T.dot( prev_state,D(self.W)))
            c = forget_gate * prev_content + input_gate * stabilized_input
            s = output_gate * T.tanh(c)

            input_gate2 = T.nnet.hard_sigmoid(
                T.dot(s,D(self.U_input_2)) + T.dot(prev_state_2,D(self.W_input_2)))
            forget_gate2 = T.nnet.hard_sigmoid(
                T.dot(s,D(self.U_forget_2)) + T.dot(prev_state_2,D(self.W_forget_2)))
            output_gate2 = T.nnet.hard_sigmoid(
                T.dot(s,D(self.U_output_2)) + T.dot(prev_state_2,D(self.W_output_2)))

            stabilized_input2 = T.tanh(T.dot(s,D(self.U_2)) + T.dot(prev_state_2,D(self.W_2)))
            c2 = forget_gate2 * prev_content_2 + input_gate2 * stabilized_input2
            s2 = output_gate2 * T.tanh(c2)



            o = T.dot(s2,self.O_w)

            return [o, s, c, s2, c2, input_gate, forget_gate, output_gate]




        [self.output,self.hidden_state,self.memory_content, _, _,self.input_gate, self.forget_gate,
         self.output_gate] , updates = theano.scan(
            forward_step,
            sequences=[X],
            truncate_gradient=-1,
            outputs_info= [None,dict(initial=T.zeros((X.shape[1],self.hidden_dim), dtype=theano.config.floatX)),
                               dict(initial=T.zeros((X.shape[1],self.hidden_dim), dtype=theano.config.floatX)),
                               dict(initial=T.zeros((X.shape[1],self.hidden_dim), dtype=theano.config.floatX)),
                               dict(initial=T.zeros((X.shape[1],self.hidden_dim), dtype=theano.config.floatX))
                               , None, None, None
                               ])




        output = T.nnet.sigmoid (self.output[-1]) #T.nnet.softmax(T.dot(self.W_out,T.sum(self.output.T,axis=1)))[0]




        params = self.params #+ self.output_params
        lambda_L1 = 0.0001
        L1_Loss = lambda_L1 * T.sum([ T.sum(abs(p)) for p in params])
        cost =  T.sum(T.nnet.binary_crossentropy(T.clip(output, 1e-7, 1.0 - 1e-7),Y)) + L1_Loss




        item_reps_in_hidden_space = T.tanh(T.dot(Y,self.ItemEmbedding))
        dists , updates = theano.scan(lambda item,hidden: T.sqrt(T.sum((item - self.hidden_state[-1]) ** 2, axis=1))
                                                          - T.sqrt(T.sum((item - hidden) ** 2))
                                      , sequences=[item_reps_in_hidden_space,self.hidden_state[-1]])

        #dists = T.reshape(dists,(dists.shape[0]*dists.shape[1],1))
        counts, updates = theano.scan(lambda item: T.ones(1),
                                     sequences=item_reps_in_hidden_space)
        #same_dists = T.sqrt(T.sum( ( item_reps_in_hidden_space - self.hidden_state[-1]) ** 2, axis=1))
        #cost_plus = (T.sum(dists) - T.sum(same_dists)) / (dists.shape[0] - same_dists.shape[0])


        cost_plus =  T.sum(T.max([T.zeros_like(dists), 0.1*T.ones_like(dists) - dists],axis=0)) #T.sum(T.max([T.zeros_like(dists),0.0*T.ones_like(dists) - dists + same_dists * T.eye(dists.shape[0])],axis=0))
        cost += cost_plus

        #self.prob_dists = theano.function([X,Y], [dists,same_dists,cost_plus])

        updates =  adam(cost,params,learning_rate=0.001) #apply_momentum(updates_sgd, params, momentum=0.9)
        self.backprop_update = theano.function([X,Y],[output,cost],updates=updates)

        self.predict = theano.function([X,Y], [output, cost])


    def calculate_accuracy(self,predictions,targets):
        corrects = 0.0
        for p,t in zip(predictions,targets):
            dd, ii = self.item_tree.query(p)
            if (self.all_items[ii] == t).all():
                corrects += 1

        return corrects / len(predictions)


class Experiment(object):
    def __init__(self,id,
                         relative_test_size,
                         number_of_concepts=3,
                         number_of_values_per_concept=3,
                         number_of_items_per_combination=3,
                         batch_size=2):

        self.id = id
        self.relative_test_size = relative_test_size
        self.number_of_concepts = number_of_concepts
        self.number_of_values_per_concept = number_of_values_per_concept
        self.number_of_items_per_combination = number_of_items_per_combination
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

        self.train_items = np.asarray(self.items[:-test_count])
        self.test_items = np.asarray(self.items[-test_count:])
        self.train_labels = np.asarray(self.labels[:-test_count])
        self.test_labels = np.asarray(self.labels[-test_count:])

        self.batch_size = batch_size
        self.number_of_training_batch = self.train_items.shape[0] // batch_size
        if (self.train_items.shape[0] % batch_size > 0):
            self.number_of_training_batch += 1

        self.number_of_testing_batch = self.test_items.shape[0] // batch_size
        if (self.test_items.shape[0] % batch_size > 0):
            self.number_of_testing_batch += 1




def do_the_exp(exp):
    lstm_listener = ImageGenerator(input_dim = exp.label_dim,
                                   hidden_dim = 256,
                                   output_dim = exp.number_of_values_per_concept*exp.number_of_concepts,
                                   dropout_rate=0.9,
                                   input_dropout_rate=.0
                                   , all_items=exp.items)

    lstm_listener.define_network()

    outputs, costs = lstm_listener.predict(np.asarray(exp.test_labels, dtype="float32").transpose(1, 0, 2),
                                           np.asarray(exp.test_items, dtype="float32")
                                           )

    number_of_epochs = 400
    exp.iteration_train_cost = []
    exp.iteration_test_cost = []

    exp.iteration_test_accuracy = []
    exp.iteration_train_accuracy = []
    for e in np.arange(number_of_epochs):
        train_items_index = np.arange(len(exp.train_labels))
        np.random.shuffle(train_items_index)
        train_costs = []
        train_predictions = []
        train_targets = []
        for k in np.arange(exp.number_of_training_batch):
            current_batch_indexes = train_items_index[
                                    k * exp.batch_size:min([(k + 1) * exp.batch_size, len(train_items_index)])]

            input_items = exp.train_labels[current_batch_indexes]
            target_label = exp.train_items[current_batch_indexes]
            output, cost = lstm_listener.backprop_update(np.asarray(input_items, dtype="float32").transpose(1, 0, 2)
                                                         , np.asarray(target_label, dtype="float32")
                                                         )
            train_costs.append(cost)
            train_predictions.extend(output)
            train_targets.extend(target_label)

        test_predictions, test_cost = lstm_listener.predict(np.asarray(exp.test_labels, dtype="float32").transpose(1, 0, 2),
                                                            np.asarray(exp.test_items, dtype="float32"))

        lstm_listener.test = False
        test_accuracy = lstm_listener.calculate_accuracy(test_predictions, exp.test_items)
        train_accuracy = lstm_listener.calculate_accuracy(train_predictions, train_targets)
        print("train_cost: " + str(np.sum(train_costs)))
        print("test_cost: " + str(test_cost))
        print("train accuracy: " + str(train_accuracy))
        print("test accuracy: " + str(test_accuracy))

        exp.iteration_train_cost.append(np.mean(train_costs))
        exp.iteration_test_cost.append(test_cost)
        exp.iteration_test_accuracy.append(test_accuracy)
        exp.iteration_train_accuracy.append(train_accuracy)
        if (test_accuracy == 1):
            break

    pickle.dump(exp, open("exp" + str(exp.id), "wb"))
    Plotting.plot_performance(exp.iteration_train_cost, exp.iteration_test_cost)
    Plotting.plot_performance(exp.iteration_train_accuracy, exp.iteration_test_accuracy)

    plt.show()


if __name__ == '__main__':
    VOCAB = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', '<\s>']
    Dic = {}
    for i in np.arange(len(VOCAB)):
        Dic[VOCAB[i]] = i
    ONE_HOT_VECS = np.eye(len(VOCAB))

    """"
    relative_test_sizes = [5, 4, 3, 2]
    for i in np.arange(len(relative_test_sizes)):
        exp = Experiment(id=i + 20,
                         relative_test_size=relative_test_sizes[i],
                         number_of_concepts=3,
                         number_of_values_per_concept=3,
                         number_of_items_per_combination=3)

        do_the_exp((exp))

    """

    """relative_test_sizes = [5, 4, 3, 2]
    for i in np.arange(len(relative_test_sizes)):
        exp = Experiment(id=i + 200,
                         relative_test_size=relative_test_sizes[i],
                         number_of_concepts=3,
                         number_of_values_per_concept=4,
                         number_of_items_per_combination=3)

        do_the_exp((exp))
    """

    """relative_test_sizes = [5, 4, 3, 2]
    for i in np.arange(3,len(relative_test_sizes)):
        exp = Experiment(id=i + 8000,
                         relative_test_size=relative_test_sizes[i],
                         number_of_concepts=3,
                         number_of_values_per_concept=4,
                         number_of_items_per_combination=3)

        do_the_exp((exp))

    plt.show()"""

    relative_test_sizes = [5, 4, 3, 2]
    for i in np.arange(3,len(relative_test_sizes)):
        exp = Experiment(id=i + 404010,
                         relative_test_size=relative_test_sizes[i],
                         number_of_concepts=4,
                         number_of_values_per_concept=10,
                         number_of_items_per_combination=4)

        do_the_exp((exp))

    plt.show()

    """exp = pickle.load(open("exp8001", "rb"))
    Plotting.plot_performance(exp.iteration_train_cost, exp.iteration_test_cost)
    Plotting.plot_performance(exp.iteration_train_accuracy, exp.iteration_test_accuracy)

    plt.show()
    """


    #  0.2 0.4 0.8


   # best last: 0.2

   #5200 0.0

   #7000 batch size = 2
  # 8000 batch size = 2, without breaks


