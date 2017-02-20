
import numpy as np


from AbstractTalker_backprop_1of3 import *
from AbstractComprehender_backprop_1of3 import *
from generate_data import *


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

    lstm_talker = TalkerLSTM(image_rep_dim=number_of_concepts * number_of_values_per_concept * number_of_items_per_combination,
                        lstm_input_dim=100,  # len(VOCAB),
                               lstm_hidden_dim=128,
                        output_dim=label_dim,
                        learning_rate=0.0005,
                        dropout_rate=0.5,
                        input_dropout_rate=0.0,
                        end_of_sentence_label_max=np.argmax(dg.labels[0][-1]))

    lstm_talker.define_network()

    lstm_listener = ListernerLSTM(input_dim =number_of_concepts * number_of_values_per_concept * number_of_items_per_combination + label_dim,
                                  hidden_dim = 128,
                                  output_dim = number_of_items_per_combination,
                                  dropout_rate=0.5,
                                  input_dropout_rate=0
                                  )
    lstm_listener.define_network()

    print("Training")


    game_iterations = 10000
    for e in np.arange(game_iterations):
        for k in np.arange(len(items)):
            state = ""
            print(str(k)+" "+get_string(labels[k],VOCAB,np.argmax(dg.labels[0][-1])))
            while state != "Succeed!":
                item = []
                shuffled_index = []
                for s in np.arange(number_of_items_per_combination):
                    item.append(items[k][s*(number_of_values_per_concept*number_of_concepts):(s+1)*(number_of_values_per_concept*number_of_concepts)])
                    shuffled_index.append(s)

                np.random.shuffle(shuffled_index)
                shuffled_item = []
                for i in shuffled_index:
                    shuffled_item.append(item[i])
                listener_target_label = np.eye(number_of_items_per_combination,dtype="float32")[shuffled_index.index(0)]

                [talker_raw_sequence] = lstm_talker.predict(np.asarray(items[k],dtype="float32"))


                if np.argmax(talker_raw_sequence[-1]) != np.argmax(dg.labels[0][-1]):
                    talker_raw_sequence = np.concatenate((talker_raw_sequence,dg.labels[-1]),axis=0)
                talker_onehot_sequence = np.asarray(get_probabilistic_sequence(talker_raw_sequence,np.argmax(dg.labels[0][-1])),dtype="float32")

                listener_shuffled_items = np.concatenate(tuple(shuffled_item),axis=0)
                listener_input_items = np.asarray([np.concatenate((listener_shuffled_items,word),axis=0) for word in talker_onehot_sequence], dtype="float32")

                listener_selection = lstm_listener.predict(listener_input_items)

                update_prob = np.random.uniform(low=0.0, high=1.0, size=(2,))


                if update_prob[0] > 0.5:
                    output_listener,cost_listener = lstm_listener.backprop_update(np.asarray(listener_input_items,dtype="float32"),listener_target_label)
                else:
                    output_listener,cost_listener = lstm_listener.get_output_without_update(np.asarray(listener_input_items,dtype="float32"),listener_target_label)

                if update_prob[1] > 0.5:
                    output_talker, cost_talker = lstm_talker.backprop_update(np.asarray(listener_shuffled_items,dtype="float32"),talker_onehot_sequence)
                else:
                    output_talker, cost_talker = lstm_talker.get_output_without_update(np.asarray(listener_shuffled_items,dtype="float32"),talker_onehot_sequence)


                if  np.argmax(listener_selection) == shuffled_index.index(0):
                        state = "Succeed!"
                else:
                        state = "Failed"

                print(state +" "+"listener cost: "+str(cost_listener)+"talker cost"+str(cost_talker)+"sequence:"+get_string(talker_onehot_sequence,VOCAB,np.argmax(dg.labels[0][-1])))

        #print listener
        #the tree options --> talker combination - listerner combination
        #listener output
        #loss
        #No dropout for listener
