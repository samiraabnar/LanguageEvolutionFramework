import numpy as np



def vector_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

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

