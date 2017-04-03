import numpy as np
import pickle
import matplotlib.pyplot as plt

from generate_data import *
from ImageGenerator import *
from CaptionGenerator_InputLoop import *
from Util import *

VOCAB = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M','N','O','P','Q','R','S','T,','U','V','W','X','Y','Z']
Dic = {}
for i in np.arange(len(VOCAB)):
    Dic[VOCAB[i]] = i
ONE_HOT_VECS = np.eye(len(VOCAB))

import sys
sys.setrecursionlimit(2000)

class SingleItemCommunicationEnv(object):
    def __init__(self,exp):
        self.exp= exp
        self.talker_model_params = {}

        self.talker_model_params["image_rep_dim"] = self.exp.number_of_concepts * self.exp.number_of_values_per_concept
        self.talker_model_params["lstm_input_dim"] = 256
        self.talker_model_params["lstm_hidden_dim"] = 256
        self.talker_model_params["output_dim"] = exp.label_dim
        self.talker_model_params["learning_rate"] = 0.005
        self.talker_model_params["dropout_rate"] = 0.9
        self.talker_model_params["input_dropout_rate"] = 0.2
        self.talker_model_params["end_of_sentence_label_max"] = self.exp.last_index



        self.talker = CaptionGenerator(image_rep_dim=self.talker_model_params["image_rep_dim"],
                                  lstm_input_dim=self.talker_model_params["lstm_input_dim"],  # len(VOCAB),
                                  lstm_hidden_dim=self.talker_model_params["lstm_hidden_dim"],
                                  output_dim=self.talker_model_params["output_dim"],
                                  learning_rate=self.talker_model_params["learning_rate"],
                                  dropout_rate=self.talker_model_params["dropout_rate"],
                                  input_dropout_rate=self.talker_model_params["input_dropout_rate"],
                                  end_of_sentence_label_max=self.talker_model_params["end_of_sentence_label_max"])
        self.talker.define_network()

        self.listener_model_params = {}
        self.listener_model_params["input_dim"] = self.exp.label_dim
        self.listener_model_params["hidden_dim"] = 256
        self.listener_model_params["output_dim"] = self.exp.number_of_values_per_concept * self.exp.number_of_concepts
        self.listener_model_params["dropout_rate"] = 0.9
        self.listener_model_params["input_dropout_rate"] = 0.0

        self.listener = ImageGenerator(input_dim=self.listener_model_params["input_dim"],
                                       hidden_dim=self.listener_model_params["hidden_dim"],
                                       output_dim=self.listener_model_params["output_dim"],
                                       dropout_rate=self.listener_model_params["dropout_rate"],
                                       input_dropout_rate=self.listener_model_params["input_dropout_rate"]
                                       , all_items=self.exp.train_items)

        self.listener.define_network()

    def communication_step(self,item):
        [talker_description] = self.talker.describe(item)
        guessed_item, guessed_item_index = self.listener.retrieve_image(talker_description)


        [cost_talker] = self.talker.get_cost(guessed_item,talker_description)
        [cost_listener] = self.listener.get_cost(talker_description,item)

        return  cost_talker, cost_listener, item, talker_description, guessed_item, guessed_item_index

    def update_networks(self, item, talker_description, guessed_item):
        [talker_description_for_guessed_item] = self.talker.describe(guessed_item)

        update_prob = np.random.uniform(low=0.0, high=1.0, size=(4,))
        if update_prob[0] > 0.5:
            [talker_output, talker_cost] = self.talker.backprop_update(guessed_item, talker_description)
        if update_prob[1] > 0.5:
            [listener_output, listener_cost] = self.listener.backprop_update(talker_description, item)
        if update_prob[2] > 0.5 and (item != guessed_item).all():

            number_one_hot = \
                [np.random.choice(a=self.talker.output_dim, size=(1,),
                                  p=vector_softmax(np.ones(talker_description_for_guessed_item.shape[1]) - word)) for word in talker_description_for_guessed_item]
            eye = np.eye(self.talker.output_dim, dtype="float32")
            [talker_output, talker_cost] = self.talker.backprop_update(item,
                                                                       np.asarray([eye[n[0]] for n in number_one_hot],dtype="float32") )
        """if update_prob[3] > 0.5:
            self.talker.discrimination_update(item,talker_description_for_guessed_item)
        """
    def game(self, number_of_epochs=100, cont=False):
        self.listener.set_allitems(self.exp.train_items)
        listener_costs = []
        talker_costs = []
        success_rates = []

        for e in np.arange(number_of_epochs):
            communication_success = []
            train_items_index = np.arange(len(exp.train_items))
            np.random.shuffle(train_items_index)
            listener_costs_epoch = []
            talker_costs_epoch = []
            for k in train_items_index:
                the_item = np.asarray(exp.train_items[k], dtype="float32")
                cost_talker, cost_listener, item, talker_description, guessed_item, guessed_item_index = self.communication_step(the_item)

                """print("costs (talker, listener): " + str(cost_talker)+" "+str(cost_listener))
                print("real item, predicted_item: "+get_string(exp.train_labels[k],VOCAB=VOCAB,end_index=exp.last_index)+" "+
                                                    get_string(exp.train_labels[guessed_item_index],VOCAB=VOCAB,end_index=exp.last_index))
                print("string: "+get_string(talker_description,VOCAB=VOCAB,end_index=exp.last_index))
                """

                self.update_networks(item,talker_description,guessed_item)

                listener_costs_epoch.append(cost_listener)
                talker_costs_epoch.append(cost_talker)
                communication_success.append(guessed_item_index == k)

            listener_costs.append(np.mean(listener_costs_epoch))
            talker_costs.append(np.mean(talker_costs_epoch))
            success_rates.append(np.sum(communication_success) / len(communication_success))

        return listener_costs,talker_costs,success_rates

    def test_play_with_no_feedback(self,number_of_epochs=1):
        self.listener.set_allitems(self.exp.test_items)
        listener_costs = []
        talker_costs = []
        success_rates = []
        for e in np.arange(number_of_epochs):
            communication_success = []
            test_items_index = np.arange(len(exp.test_items))
            np.random.shuffle(test_items_index)
            listener_costs_epoch = []
            talker_costs_epoch = []
            for k in test_items_index:
                the_item = np.asarray(exp.test_items[k], dtype="float32")
                cost_talker, cost_listener, item, talker_description, guessed_item, guessed_item_index = self.communication_step(
                    the_item)

                """print("costs (talker, listener): " + str(cost_talker) + " " + str(cost_listener))
                print("real item, predicted_item: " + get_string(exp.test_labels[k], VOCAB=VOCAB,
                                                                 end_index=exp.last_index) + " " +
                      get_string(exp.test_labels[guessed_item_index], VOCAB=VOCAB, end_index=exp.last_index))
                print("string: " + get_string(talker_description, VOCAB=VOCAB, end_index=exp.last_index))
                """
                listener_costs_epoch.append(cost_listener)
                talker_costs_epoch.append(cost_talker)
                communication_success.append(guessed_item_index == k)

            listener_costs.append(np.mean(listener_costs_epoch))
            talker_costs.append(np.mean(talker_costs_epoch))
            success_rates.append(np.sum(communication_success) / len(communication_success))

        return listener_costs,talker_costs,success_rates

    def save(self):
        print("Saving Env...")
        pickle.dump(self.exp, open("single_item_communication_exp" + str(self.exp.id), "wb"))
        pickle.dump(self.talker, open("single_item_communication_talker" + str(self.exp.id), "wb"))
        pickle.dump(self.listener, open("single_item_communication_listener" + str(self.exp.id), "wb"))

    def analyse_vocab(self):
        vocab = {}
        new_vocab = {}

        for k in np.arange(len(exp.train_items)):
            [caption] = self.talker.describe(np.asarray(self.exp.train_items[k], dtype="float32"))
            string_caption = get_string(caption, VOCAB, self.exp.last_index)

            if string_caption not in vocab.keys():
                vocab[string_caption] = []

            vocab[string_caption].append(get_string(self.exp.train_labels[k], VOCAB, exp.last_index))


        for k in np.arange(len(exp.test_items)):
            [caption] = self.talker.describe(np.asarray(self.exp.test_items[k], dtype="float32"))
            string_caption = get_string(caption, VOCAB, self.exp.last_index)

            if string_caption not in vocab.keys():
                if string_caption not in new_vocab.keys():
                    new_vocab[string_caption] = []
                new_vocab[string_caption].append(get_string(self.exp.test_labels[k], VOCAB, self.exp.last_index))

        print("Vocab Length:" + str(len(vocab.keys())))
        print("New Vocab Length:" + str(len(new_vocab.keys())))

        print(vocab)
        print(new_vocab)

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




if __name__ == '__main__':
    relative_test_sizes = [2,3,4,5]
    i = 1
    exp = Experiment(id=i + 1 + 9900,
                         relative_test_size=relative_test_sizes[i],
                         number_of_concepts=3,
                         number_of_values_per_concept=5,
                         number_of_items_per_combination=3,
                         number_of_epochs=20
                     )

    exp.prepare_data()
    env = SingleItemCommunicationEnv(exp)
    print("Environment is initialized :)")

    print("Play ...")
    test_listener_costs, test_talker_costs, test_success_rates = [], [], []
    train_listener_costs, train_talker_costs, train_success_rates = [],[],[]

    for turn in np.arange(50):
        listener_costs, talker_costs, success_rates = env.game(exp.number_of_epochs)
        listener_costs, talker_costs, success_rates = np.mean(listener_costs), np.mean(talker_costs), np.mean(success_rates)
        print("train: "+ str(listener_costs), str(talker_costs), str(success_rates))

        train_listener_costs.append(listener_costs)
        train_talker_costs.append(talker_costs)
        train_success_rates.append(success_rates)

        listener_costs, talker_costs, success_rates = env.test_play_with_no_feedback()
        print("test: "+ str(listener_costs), str(talker_costs), str(success_rates))

        test_listener_costs.extend(listener_costs)
        test_talker_costs.extend(talker_costs)
        test_success_rates.extend(success_rates)

        env.analyse_vocab()


    Plotting.plot_performance(train_listener_costs, train_listener_costs)
    Plotting.plot_performance(train_talker_costs, train_talker_costs)
    Plotting.plot_performance(train_success_rates, train_success_rates)
    Plotting.plot_performance(test_listener_costs, test_listener_costs)
    Plotting.plot_performance(test_talker_costs, test_talker_costs)
    Plotting.plot_performance(test_success_rates, test_success_rates)

    output_embeddings = []
    text_train = []
    labels = []
    for k in np.arange(len(exp.train_items)):
        [embedding] = env.talker.get_last_hidden_state(np.asarray(exp.train_items[k], dtype="float32"))
        [caption] = env.talker.describe(np.asarray(exp.train_items[k], dtype="float32"))
                                         #  np.asarray(exp.train_labels[k], dtype="float32"))
        output_embeddings.append(embedding)
        text_train.append(10)
        labels.append(get_string(exp.train_labels[k][:-1], VOCAB, exp.last_index)+"---"+
                      get_string(caption, VOCAB, exp.last_index))

    for k in np.arange(len(exp.test_items)):
        [embedding] = env.talker.get_last_hidden_state(np.asarray(exp.test_items[k], dtype="float32"))
                                          # np.asarray(exp.test_labels[k], dtype="float32"))
        [caption] = env.talker.describe(np.asarray(exp.train_items[k], dtype="float32"))
        output_embeddings.append(embedding)
        text_train.append(30)
        labels.append(get_string(exp.test_labels[k][:-1], VOCAB, exp.last_index)+"---"+
                      get_string(caption, VOCAB, exp.last_index))

    Plotting.plot_distribution_t_SNE(np.asarray(output_embeddings),
                                     text_train
                                     # np.argmax(exp.labels[:,:,2], axis=1) + 10* np.argmax(exp.train_labels[:,:,1], axis=1)  + 100* np.argmax(exp.train_labels[:,:,0], axis=1)# [word_set.index(word) for word in words]
                                     , labels)


    plt.show()

    env.save()


    # + 500 --> different update for talker
    # + 600 --> different update for talker--> 20 epochs each time.
    # + 700 --> different update for talker--> 40 epochs each time.
    # + 800 --> different update for talker--> 40 epochs each time + 3rd update only of wrong.




