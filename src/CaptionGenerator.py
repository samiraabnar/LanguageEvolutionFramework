import theano.tensor as T
import numpy as np

import sys
import pickle

sys.setrecursionlimit(1000)

from theanolm.network.weightfunctions import *
from lasagne.updates import adam
import theano.tensor.shared_randomstreams

from generate_data import *
from Util import *

class CaptionGenerator(object):
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
        self.rng = theano.tensor.shared_randomstreams.RandomStreams(self.random_state.randint(999999))
        self.end_of_sentence_label_max = theano.shared(value=end_of_sentence_label_max, name="EndSymbol", borrow="True")

        self.test = False
        #self.image_reader = VGG_16('vgg16_weights.h5')
        #self.image_reader.model.compile(optimizer='adam', loss='categorical_crossentropy')


    def init_lstm_weights(self):
      #  WordEmbedding = random_normal_matrix((self.lstm_input_dim // 2, self.output_dim))
        ImageEmbedding = random_normal_matrix((self.lstm_input_dim , self.image_rep_dim))

       # self.WordEmbedding = theano.shared(value=WordEmbedding, name="WordEmbedding", borrow="True")
        self.ImageEmbedding = theano.shared(value=ImageEmbedding, name="ImageEmbedding", borrow="True")

        U_input = random_normal_matrix((self.lstm_hidden_dim, self.lstm_input_dim),scale=1.0)

        U_forget = random_normal_matrix((self.lstm_hidden_dim, self.lstm_input_dim),scale=1.0)

        U_output = random_normal_matrix((self.lstm_hidden_dim, self.lstm_input_dim),scale=1.0)

        W_input = random_normal_matrix((self.lstm_hidden_dim, self.lstm_hidden_dim),scale=1.0)

        W_forget = random_normal_matrix((self.lstm_hidden_dim, self.lstm_hidden_dim),scale=1.0)

        W_output = random_normal_matrix((self.lstm_hidden_dim, self.lstm_hidden_dim),scale=1.0)

        U = random_normal_matrix((self.lstm_hidden_dim, self.lstm_input_dim),scale=1.0)

        W = random_normal_matrix((self.lstm_hidden_dim, self.lstm_hidden_dim),scale=1.0)

        self.W = theano.shared(value=W, name="W" , borrow="True")
        self.U = theano.shared(value=U, name="U" , borrow="True")

        self.W_input = theano.shared(value=W_input, name="W_input" , borrow="True")
        self.U_input = theano.shared(value=U_input, name="U_input" , borrow="True")

        self.W_output = theano.shared(value=W_output, name="W_output" , borrow="True")
        self.U_output = theano.shared(value=U_output, name="U_output" , borrow="True")

        self.W_forget = theano.shared(value=W_forget, name="W_forget" , borrow="True")
        self.U_forget = theano.shared(value=U_forget, name="U_forget" , borrow="True")


        U_input_2 = random_normal_matrix((self.lstm_hidden_dim, self.lstm_hidden_dim),scale=1.0)

        U_forget_2 = random_normal_matrix((self.lstm_hidden_dim, self.lstm_hidden_dim),scale=1.0)

        U_output_2 = random_normal_matrix((self.lstm_hidden_dim, self.lstm_hidden_dim),scale=1.0)

        W_input_2 = random_normal_matrix((self.lstm_hidden_dim, self.lstm_hidden_dim),scale=1.0)

        W_forget_2 = random_normal_matrix((self.lstm_hidden_dim, self.lstm_hidden_dim),scale=1.0)

        W_output_2 = random_normal_matrix((self.lstm_hidden_dim, self.lstm_hidden_dim),scale=1.0)

        U_2 = random_normal_matrix((self.lstm_hidden_dim, self.lstm_hidden_dim),scale=1.0)

        W_2 = random_normal_matrix((self.lstm_hidden_dim, self.lstm_hidden_dim),scale=1.0)



        self.W_2 = theano.shared(value=W_2, name="W_2" , borrow="True")
        self.U_2 = theano.shared(value=U_2, name="U_2" , borrow="True")

        self.W_input_2 = theano.shared(value=W_input_2, name="W_input_2" , borrow="True")
        self.U_input_2 = theano.shared(value=U_input_2, name="U_input_2" , borrow="True")

        self.W_output_2 = theano.shared(value=W_output_2, name="W_output_2" , borrow="True")
        self.U_output_2 = theano.shared(value=U_output_2, name="U_output_2" , borrow="True")

        self.W_forget_2 = theano.shared(value=W_forget_2, name="W_forget_2" , borrow="True")
        self.U_forget_2 = theano.shared(value=U_forget_2, name="U_forget_2" , borrow="True")





        O_w = random_normal_matrix((self.output_dim, self.lstm_hidden_dim),scale=1.0)
        self.O_w = theano.shared(value=O_w, name="O_w" , borrow="True")


        self.params = [self.U_input, self.U_forget, self.U_output, self.W_input, self.W_forget, self.W_output,
                       self.U, self.W,
                       self.U_input_2, self.U_forget_2, self.U_output_2, self.W_input_2, self.W_forget_2, self.W_output_2,
                       self.U_2, self.W_2,
                       self.O_w,  self.ImageEmbedding
                       #self.WordEmbedding,
                       ]


    def define_network(self):

        Y = T.matrix('output')
        H = T.vector('init_state')

        O = T.vector('other_item')


        self.init_lstm_weights()

        def D(x):
            if self.dropout_rate == 0:
                return x
            else:
                retain_prob = 1 - self.dropout_rate
                return x * np.random.binomial(1, retain_prob, self.lstm_hidden_dim).astype(dtype=np.float32)

        def Input_D(x):
            if self.input_dropout_rate == 0 or self.test == True:
                return x
            else:
                retain_prob = 1 - self.dropout_rate
                return x * np.random.binomial(1, retain_prob, self.lstm_input_dim).astype(dtype=np.float32)


        def forward_step(viz_o,x_t, prev_state, prev_content, prev_state_2, prev_content_2):
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


            #sample_sdv_index = self.rng.choice(size=(1,), a=self.output_dim,p=output_sdv)[0]
            viz_o_2 = T.nnet.softmax(T.dot(self.O_w, s2))[0]
            visible_output_index = self.rng.choice(size=(1,), a=self.output_dim,p=viz_o_2)[0]
            the_output = T.eye(self.output_dim,dtype=theano.config.floatX)[visible_output_index]

            #o = T.dot(self.WordEmbedding,viz_o)
            next_input = T.dot(self.ImageEmbedding, H)
            # T.concatenate([T.dot(self.ImageEmbedding, H), T.dot(self.WordEmbedding, viz_o_2)], axis=0)

            return [viz_o_2,next_input, s, c, s2, c2, input_gate, forget_gate, output_gate,the_output], \
                   theano.scan_module.until(
                       T.eq(visible_output_index,self.end_of_sentence_label_max))



        [self.cost_output, _,_,_,self.hidden_state, self.memory_content, self.input_gate, self.forget_gate,
         self.output_gate, self.output], scan_updates = theano.scan(
            forward_step,
            #sequences=[X],
            truncate_gradient=-1,
            n_steps= 4,
            outputs_info=[dict(initial=T.zeros(self.output_dim, dtype=theano.config.floatX)),
                          dict(initial = T.dot(self.ImageEmbedding,H)), #, T.dot(self.WordEmbedding,T.zeros(self.output_dim))])),
                          dict(initial= T.zeros(self.lstm_hidden_dim, dtype=theano.config.floatX)),
                          dict(initial=T.zeros(self.lstm_hidden_dim, dtype=theano.config.floatX)),
                          dict(initial= T.zeros(self.lstm_hidden_dim, dtype=theano.config.floatX)),
                          dict(initial=T.zeros(self.lstm_hidden_dim, dtype=theano.config.floatX))
                , None, None, None, None
                          ])




        params = self.params

        length = T.max([Y.shape[0], self.output.shape[0]])



        padded_cost_output = T.zeros((length, Y.shape[1]))
        padded_cost_output = T.set_subtensor(padded_cost_output[0:self.cost_output.shape[0], :], self.cost_output)
        padded_cost_output = T.set_subtensor(padded_cost_output[self.cost_output.shape[0]:, :],
                                        T.zeros(self.output_dim, "float32"))

        padded_output = T.zeros((length, Y.shape[1]))
        padded_output = T.set_subtensor(padded_output[0:self.output.shape[0], :], self.output)
        padded_output = T.set_subtensor(padded_output[self.output.shape[0]:, :],
                                             T.zeros(self.output_dim, "float32"))

        padded_Y = T.zeros((length, Y.shape[1]))
        padded_Y = T.set_subtensor(padded_Y[0:Y.shape[0], :], Y)
        padded_Y = T.set_subtensor(padded_Y[Y.shape[0]:, :], T.zeros(self.output_dim,"float32"))

        #padded_cost_output = padded_cost_output * (padded_output + padded_Y)

        padded_Y = padded_Y + (1 - (padded_output + padded_Y))  *  padded_cost_output






        lambda_1 = pow(10, -(T.log((self.lstm_hidden_dim + self.lstm_input_dim) / 2) / T.log(10)) - 2)
        lambda_2 = pow(10, -(T.log(self.lstm_hidden_dim) / T.log(10)) * 2 - 2)


        lambda_L1 = 0.001
        L1_Loss = lambda_L1 * T.sum([T.sum(abs(p)) for p in params])
        L2_Loss = lambda_2 * T.sum([T.sum(p ** 2) for p in params])
        cost = T.sum(T.nnet.categorical_crossentropy(padded_cost_output + 0.00000001, padded_Y)) + L1_Loss + L2_Loss
        #   +   lambda_1 * sum([T.sum(param ** 2) for param in params])
        #+ lambda_2 * sum([T.sum(abs(param) > 0) for param in params])
        updates = adam(cost, params, learning_rate=self.learning_rate)




        self.backprop_update = theano.function([H, Y], [self.output,cost], updates=updates+scan_updates)
        self.predict = theano.function([H,Y], [self.output,cost], updates=scan_updates)
        self.get_last_hidden_state = theano.function([H],[self.hidden_state[-1]],updates=scan_updates)
        self.describe = theano.function([H], [self.output], updates=scan_updates)
        self.get_cost = theano.function([H, Y], [cost],updates=scan_updates)
        #backprop_update_with_feedback_negative = theano.function([H, Y], [cost3], updates=feedback_updates_neg)

        self.get_image_embedding = theano.function([H],[T.dot(self.ImageEmbedding,H)])

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
        #if np.argmax(list_of_vec[i]) != end_index:
        #    a = list(list_of_vec[i])
        #    a[end_index] = 0
        #    a = a /sum(a)
        #    str += " "+VOCAB[np.random.choice(len(a), 1,p=a)[0]]
        #else:
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



class Experiment(object):
    def __init__(self,id,relative_test_size,number_of_concepts,number_of_values_per_concept,number_of_items_per_combination,number_of_epochs):
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
        self.number_of_epochs = number_of_epochs


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


    def do(self):
        self.model_params = {}
        self.model_params["image_rep_dim"] = exp.number_of_concepts * exp.number_of_values_per_concept
        self.model_params["lstm_input_dim"] = 256
        self.model_params["lstm_hidden_dim"] = 256
        self.model_params["output_dim"] = exp.label_dim
        self.model_params["learning_rate"] = 0.01
        self.model_params["dropout_rate"] = 0.9
        self.model_params["input_dropout_rate"] = 0.2
        self.model_params["end_of_sentence_label_max"] = exp.last_index
        talker = CaptionGenerator(image_rep_dim= self.model_params["image_rep_dim"],
                                  lstm_input_dim=self.model_params["lstm_input_dim"],  # len(VOCAB),
                            lstm_hidden_dim=self.model_params["lstm_hidden_dim"],
                                  output_dim=self.model_params["output_dim"],
                                  learning_rate=self.model_params["learning_rate"],
                                  dropout_rate=self.model_params["dropout_rate"],
                                  input_dropout_rate=self.model_params["input_dropout_rate"],
                                  end_of_sentence_label_max=self.model_params["end_of_sentence_label_max"])

        talker.define_network()


        print("Network defined :)")

        test_cost = []
        train_cost = []
        for e in np.arange(self.number_of_epochs):
            train_items_index = np.arange(len(exp.train_items))
            np.random.shuffle(train_items_index)
            train_cost_ = 0
            for k in train_items_index:
                output, cost = talker.backprop_update(np.asarray(exp.train_items[k], dtype="float32"),
                                                      np.asarray(exp.train_labels[k], dtype="float32"))
                # print(str(k)+": "+get_string(train_labels[k],VOCAB,last_index)+" : "+get_string(output,VOCAB,last_index))
                train_cost_ += cost

            train_cost.append(train_cost_ / len(train_items_index))

            talker.test = True
            test_cost_ = 0
            for k in np.arange(len(exp.test_items)):
                output, cost = talker.predict(np.asarray(exp.test_items[k], dtype="float32"),
                                              np.asarray(exp.test_labels[k], dtype="float32"))
                # print(str(k)+": "+get_string(train_labels[k],VOCAB,last_index)+" : "+get_string(output,VOCAB,last_index))
                test_cost_ += cost

            test_cost.append(test_cost_ / len(exp.test_items))
            talker.test = False

            print(str(e) + ": " + str(train_cost[-1]) + '   ' + str(test_cost[-1]))

            """if (e+1) % 100 == 0:
                Plotting.plot_performance(train_cost, test_cost)
            """

        Plotting.plot_performance(train_cost, test_cost)

        print("next plot...")
        embeddings = []
        text_train = []
        labels = []
        for k in np.arange(len(exp.train_items)):
            [embedding] = talker.get_image_embedding(np.asarray(exp.train_items[k], dtype="float32"))
            embeddings.append(embedding)
            text_train.append(10)
            labels.append(get_string(exp.train_labels[k], VOCAB, exp.last_index))

        for k in np.arange(len(exp.test_items)):
            [embedding] = talker.get_image_embedding(np.asarray(exp.test_items[k], dtype="float32"))
            embeddings.append(embedding)
            text_train.append(30)
            labels.append(get_string(exp.test_labels[k], VOCAB, exp.last_index))

        Plotting.plot_distribution_t_SNE(np.asarray(embeddings),
                                         text_train
                                         # np.argmax(exp.labels[:,:,2], axis=1) + 10* np.argmax(exp.train_labels[:,:,1], axis=1)  + 100* np.argmax(exp.train_labels[:,:,0], axis=1)# [word_set.index(word) for word in words]
                                         , labels)

        print("next plot...")
        output_embeddings = []
        text_train = []
        labels = []
        for k in np.arange(len(exp.train_items)):
            [embedding] = talker.get_last_hidden_state(np.asarray(exp.train_items[k], dtype="float32"))
                                            #  np.asarray(exp.train_labels[k], dtype="float32"))
            output_embeddings.append(embedding)
            text_train.append(10)
            labels.append(get_string(exp.train_labels[k], VOCAB, exp.last_index))

        for k in np.arange(len(exp.test_items)):
            [embedding] = talker.get_last_hidden_state(np.asarray(exp.test_items[k], dtype="float32"))
                                             # np.asarray(exp.test_labels[k], dtype="float32"))
            output_embeddings.append(embedding)
            text_train.append(30)
            labels.append(get_string(exp.test_labels[k], VOCAB, exp.last_index))

        Plotting.plot_distribution_t_SNE(np.asarray(output_embeddings),
                                         text_train
                                         # np.argmax(exp.labels[:,:,2], axis=1) + 10* np.argmax(exp.train_labels[:,:,1], axis=1)  + 100* np.argmax(exp.train_labels[:,:,0], axis=1)# [word_set.index(word) for word in words]
                                         , labels)

        plt.show()

        pickle.dump(output_embeddings, open("output_embeddings" + str(self.id), "wb"))
        pickle.dump(exp, open("exp_caption_generator_"+str(self.id), "wb"))



if __name__ == '__main__':

    VOCAB = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L','M']
    Dic = {}
    for i in np.arange(len(VOCAB)):
        Dic[VOCAB[i]] = i
    ONE_HOT_VECS = np.eye(len(VOCAB))


    i= 0
    relative_test_sizes = [5,4,3,2]
    for i in np.arange(len(relative_test_sizes)):
        exp = Experiment(id=i + 1,
                         relative_test_size=relative_test_sizes[i],
                         number_of_concepts=3,
                         number_of_values_per_concept=4,
                         number_of_items_per_combination=3,
                         number_of_epochs=250
                         )

        exp.prepare_data()
        exp.do()

