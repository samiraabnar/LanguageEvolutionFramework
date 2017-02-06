import numpy as np
from enum import Enum
import itertools

class DataMode(Enum):
    ONE_HOT = 1


class DataSetGenerator(object):
    def __init__(self,number_of_independent_concepts=2,number_of_values_per_independent_concept=3,mode=DataMode.ONE_HOT):
        self.number_of_independent_concepts = number_of_independent_concepts
        self.number_of_values_per_independent_concept = number_of_values_per_independent_concept

        self.items = np.asarray([])
        self.labels = np.asarray([])
        self.combined_items = []
        self.combined_labels = []

    def code_independent_concepts(self):
        concept_indexes = np.arange(self.number_of_independent_concepts)
        concept_value_indexes = np.arange(
            self.number_of_independent_concepts * self.number_of_values_per_independent_concept)
        independent_concept_value_labels = np.eye(len(concept_value_indexes)+1)
        np.random.shuffle(independent_concept_value_labels)
        independent_concepts_values = np.zeros(
            (self.number_of_independent_concepts, self.number_of_values_per_independent_concept,
             self.number_of_values_per_independent_concept))
        for concept_index in concept_indexes:
            hot_indexes = np.arange(self.number_of_values_per_independent_concept)
            np.random.shuffle(hot_indexes)
            for  value_index in np.arange(self.number_of_values_per_independent_concept):
                independent_concepts_values[concept_index][value_index][hot_indexes[value_index]] = 1

            assert ((np.sum(independent_concepts_values[concept_index], axis=0) ==
                    np.ones(self.number_of_values_per_independent_concept)).all()), "concept values are not valid"
        concat_concept_values_all_combinations = []
        concat_concept_values_all_combinations_labels = []
        return concat_concept_values_all_combinations, concat_concept_values_all_combinations_labels, independent_concept_value_labels, independent_concepts_values

    def generate_data(self):

        concat_concept_values_all_combinations, \
        concat_concept_values_all_combinations_labels, \
        independent_concept_value_labels, \
        independent_concepts_values = self.code_independent_concepts()

        concat_concept_values_all_combinations_indexes = list(itertools.product(np.arange(self.number_of_values_per_independent_concept), repeat=self.number_of_independent_concepts))
        for concat_concept_value_indexes in concat_concept_values_all_combinations_indexes:
            combined_concept_value = [independent_concepts_values[k][concat_concept_value_indexes[k]] for k in np.arange(self.number_of_independent_concepts)]
            combined_concept_value_label = [independent_concept_value_labels[k][concat_concept_value_indexes[k]] for k in np.arange(self.number_of_independent_concepts)]


            concated_combined_concept_value = np.concatenate(tuple(combined_concept_value), axis=0)
            concated_combined_concept_value_label = np.concatenate(tuple(combined_concept_value_label), axis=0)


            concat_concept_values_all_combinations.append(concated_combined_concept_value)
            concat_concept_values_all_combinations_labels.append(concated_combined_concept_value_label)



        self.items = concat_concept_values_all_combinations
        self.labels = concat_concept_values_all_combinations_labels


    def generate_data_with_marginal_labels(self,number_of_items_per_label=3,core_item=0):
        self.combined_items = []
        self.combined_labels = []

        concat_concept_values_all_combinations, \
        concat_concept_values_all_combinations_labels, \
        independent_concept_value_labels, \
        independent_concepts_values = self.code_independent_concepts()

        concat_concept_values_all_combinations_indexes = list(
            itertools.product(np.arange(self.number_of_values_per_independent_concept),
                              repeat=self.number_of_independent_concepts))
        for concat_concept_value_indexes in concat_concept_values_all_combinations_indexes:
            combined_concept_value = [independent_concepts_values[k][concat_concept_value_indexes[k]] for k in
                                      np.arange(self.number_of_independent_concepts)]
            combined_concept_value_label = [independent_concept_value_labels[k*self.number_of_values_per_independent_concept+ concat_concept_value_indexes[k]] for k
                                            in np.arange(self.number_of_independent_concepts)]

            combined_concept_value_label.append(independent_concept_value_labels[-1])
            concated_combined_concept_value = np.concatenate(tuple(combined_concept_value), axis=0)
            concated_combined_concept_value_label = np.asarray(combined_concept_value_label)

            concat_concept_values_all_combinations.append(concated_combined_concept_value)
            concat_concept_values_all_combinations_labels.append(concated_combined_concept_value_label)

        self.items = concat_concept_values_all_combinations
        self.labels = concat_concept_values_all_combinations_labels

        ordered_combination_of_items = list(
            itertools.permutations(np.arange(len(self.items)),
                              r= number_of_items_per_label))
        for combined_items_indexes in ordered_combination_of_items:
            combined_items = [ self.items[combined_items_indexes[item_index]] for item_index in np.arange(number_of_items_per_label)]

            #default sorts min to max


            remained_items = list(combined_items)
            remained_items.pop(core_item)
            remained_items = np.asarray(remained_items)

            sorted_index_index = 0
            selected_concepts_value_indexes = []
            non_zero_index_core_item = np.where(combined_items[core_item] > 0)[0]
            while remained_items.shape[0] > 0:
                count_shared_concepts = np.sum(remained_items[:,non_zero_index_core_item], axis=0)
                sorted_index = non_zero_index_core_item[np.argsort(count_shared_concepts)]


                remove_indexes = np.where(remained_items[:,sorted_index[sorted_index_index]] != 1)
                remained_items = remained_items[np.delete(np.arange(len(remained_items)),remove_indexes) ]
                selected_concepts_value_indexes.append(sorted_index[0] % self.number_of_values_per_independent_concept)


            combined_item_label = []

            for i in np.arange(len(self.labels[combined_items_indexes[core_item]])):
                if i in selected_concepts_value_indexes:
                    combined_item_label.append(self.labels[combined_items_indexes[core_item]][i])

            combined_item_label.append(independent_concept_value_labels[-1])
            concat_combined_item = np.concatenate(combined_items,axis=0)

            self.combined_items.append(concat_combined_item)
            self.combined_labels.append(combined_item_label)










    def get_items(self):
        return self.items

    def get_labels(self):
        return self.labels


if __name__ == '__main__':
    dg = DataSetGenerator(2,2)
    dg.generate_data_with_marginal_labels(number_of_items_per_label=2,core_item=0)

    for i,l in zip(dg.items,dg.labels):
        print(i, l)

    for i,l in zip(dg.combined_items,dg.combined_labels):
        print(i, l)