import random
import InduceC45
import sys
import json
import pandas as pd
import time
import numpy as np
from contextlib import redirect_stdout
from knn_implementation import print_results
import multiprocessing

def prepare_D(training_file, ground_truth_file): 
    D = pd.read_csv(training_file, sep=',', engine='c', na_filter=False, low_memory = False)
    D = D.drop(columns=['Unnamed: 0'])
    D = D.reset_index(drop=True)
    A = pd.read_csv(ground_truth_file)
    A = A.reset_index(drop=True)
    #merge D and A into one dataframe
    D = pd.merge(D, A, left_index=True, right_index=True)
    D = D.reset_index(drop=True)
   
    data_list = [row for row in D.values]

    i = list(range(len(D.columns)))
    data_attrs = dict(zip(D.columns, i))
    c = data_attrs["author"]
    attr_cards = [0] * len(data_list[0])
    attr_cards[-1] = 50

    return data_list, data_attrs, c, attr_cards

def random_forest(D, A, m, k, N, threshold, c, attr_cards):
    trees = list()
    attr_lists = []
    new_attr_cards = []
    count = 0
    while count < N:
        (d, a, new_c) = dataset_selector(D, A, m, k, c)
        curr_attr_cards = [attr_cards[value] for value in a.values()]
        new_tree = InduceC45.C45(d, a, threshold, new_c, curr_attr_cards, D)
        trees.append(new_tree)
        attr_lists.append(a)
        new_attr_cards.append(curr_attr_cards)
        count += 1
    return trees, attr_lists, new_attr_cards

def partition_data(d):
    # returns 10 sets where union of all sets = D
    sets = []
    data_left = d.copy()
    size = int(len(d) / 10)
    for i in range(9):
        curr_set = []
        for j in range(size):
            index = random.randint(0, len(data_left) - 1)
            curr_set.append(data_left[index])
            data_left.pop(index)
        sets.append(curr_set)
    sets.append(data_left)
    return sets

# decide whether c counts as 1 of the attributes, or if it is separate
def dataset_selector(D, A, m, k, c):
    # will not make more attributes or more pts
    reverse_A = dict((y, x) for x, y in A.items())
    pt_indices = list(np.random.choice(len(D), k, replace=False))
    at_indices = list(np.random.choice(len(D[0]) - 2, m, replace=False))
    at_indices.append(c)
    
    new_D = [[D[i][x] for x in at_indices] for i in pt_indices]

    new_A = dict()
    count = 0
    for i in at_indices:    #OPT
        new_A[reverse_A[i]] = i
        if i == c:
            new_c = count
        count += 1

    return new_D, new_A, new_c

# noinspection PyBroadException
def traverse_tree(d, A, tree):
    try:
        var = tree['node']['var']
    except:
        return tree['leaf']['decision']

    move = d[A[var]]
    edges = tree['node']['edges']
    for edge in edges:
        if edge['edge']['direction'] == 'le':
            take_edge = float(move) <= float(edge['edge']['value'])
        else:
            take_edge = float(move) > float(edge['edge']['value'])
        if take_edge:
            return traverse_tree(d, A, edge['edge'])
    return -1   # this shouldn't happen

def traverse_forest(d, A, trees):
    expected = [traverse_tree(d, A, trees[i]) for i in range(len(trees))]
    return max(set(expected), key=expected.count)

def classifier(A, D, c, jason, attr_cards):
    dummy_vals = dict()
    dom = InduceC45.get_domain(D, c)
    dom = sorted(dom)
    # print(dom)
    for n in range(len(dom)):
        dummy_vals[dom[n]] = n
    sz = len(dom)
    confusion_matrix = np.zeros((sz, sz), dtype = np.float32)    
    trees = jason
    for i in range(len(D)):
        expected = traverse_forest(D[i], A, trees)
        actual = D[i][c]
        confusion_matrix[dummy_vals[actual]][dummy_vals[expected]] += 1
    
    return confusion_matrix, dummy_vals

def generate_RF(stopword_stemming):
    #timer
    start = time.time()

    csv = stopword_stemming[0] + "_" + stopword_stemming[2] + "_" + str(stopword_stemming[3]) + ".csv"
    m = 500
    k = stopword_stemming[1]
    N = stopword_stemming[4]
    threshold = 0.1
  
    # validation for lab5
    D, A, c, cardinalities = prepare_D(csv, "groundtruth_overall.csv")
    forest, attr_lists, card_lists = random_forest(D, A, m, k, N, threshold, c, cardinalities)
    json_objs = []
    for tree in forest:
        json_str = InduceC45.create_json_str(csv, tree)
        json_obj = json.loads(json_str)
        json_objs.append(json_obj)

    # chang
    confusion_matrix, rowColDict = classifier(A, D, c, json_objs, cardinalities)
    row_col_names = list(rowColDict.keys())

    print_results(confusion_matrix, row_col_names)

    # timer
    end = time.time()
    print("Time Taken: " + str(end - start))


def generate_outputs(stopword_stemming):
    with open(stopword_stemming[0] + "_" + stopword_stemming[2] + "_" + str(stopword_stemming[3]) + "_k=" + str(stopword_stemming[1]) + "_" + "trees=" + str(stopword_stemming[4]) + ".txt", 'w') as f:
        with redirect_stdout(f):
            generate_RF(stopword_stemming)


if __name__ == "__main__":
    words = ["overall_tf_idf"]
    k_list = [10, 15, 20]
    num_tress = [4000, 2667, 2000]
    stopword_type = ["long", "short", "empty", "onix"]
    stemming = [True, False]

    #merge stopword_type and stemming into one list list of tuples list comprehension
    stopword_stemming = [(words[0], z, stopword_type[i], stemming[j], w) for i in range(len(stopword_type)) for j in range(len(stemming)) for w, z in zip(k_list, num_tress)]

    p = multiprocessing.Pool(processes=16)
    p.map(generate_outputs, stopword_stemming)
    p.close()
    p.join()


