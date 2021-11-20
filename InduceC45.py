from operator import itemgetter

import pandas as pd
import math
import json
import sys
import time

class Node:
    def __init__(self, label):
        self.label = label
        self.children = []
        self.p = None
        self.alpha = None
        # children has structure: list of tuples
        # each tuple = (edge_label, child_tree)


def categorical(attr_cards, i):
    return False    # no categorical
    if attr_cards[i] == 0 or attr_cards[i] == '0' or attr_cards[i] == 0.0 or attr_cards[i] == "0.0":
        return False
    return True


def parse_restrictions(filename):
    f = open(filename)
    line = f.read()
    attrs = []
    i = 0
    for char in line:
        if char == '1':
            attrs.append(i)
        i += 1
    return attrs


def get_domain(D, i):
    """ Returns all possible values of A_i given D and i """
    return list(set([curr[i] for curr in D]))


def entropy(D, c):
    ''' calculates and returns entropy of D with respect to c,
    assuming c is the last attribute stored '''
    sum = 0
    dom_c = get_domain(D, c)
    for val in dom_c:
        count = 0
        for curr in D:
            if curr[c] is val:
                count += 1
        sum += (count / len(D)) * math.log(count / len(D), 2)
    return -sum


def entropy_ai(D, i, c):
    """calculates entropy_Ai of D, given D and i"""
    dom_ai = get_domain(D, i)
    sum = 0
    for val in dom_ai:
        D_j = [curr for curr in D  if curr[i] is val]
        to_add = len(D_j) / len(D)
        to_add *= entropy(D_j, c)
        sum += to_add
    return sum

def entropy_ai_alpha(D, ai, alpha, c):
    """calculates entropy_ai of D on split alpha"""
    d_plus, d_minus = split_on_alpha(D, ai, alpha)
    return (float(len(d_minus) / len(D)) * entropy(d_minus, c)) + float((len(d_plus) / len(D)) * entropy(d_plus, c))

def split_on_alpha(D, i, alpha):
    d_plus = []
    d_minus = []
    [d_minus.append(curr) if curr[i] <= alpha else d_plus.append(curr) for curr in D]
    return d_plus, d_minus

def find_most_frequent_label(D):
    """calculates most common label in entire dataset"""
    frequent = None
    max = 0
    for i in range(len(D[0])):
        (frq, count) = find_most_frequent_label_col(D, i)
        if count > max:
            max = count
            frequent = frq
    return frequent


def find_most_frequent_label_col(D, i):
    """calculates (most common label, count) for column i"""
    dom = get_domain(D, i)
    frequent = None
    max = 0
    for d in dom:
        count = 0
        for row in D:
            if row[i] == d:
                count += 1
        if count > max:
            max = count
            frequent = d
    return frequent, max


def select_splitting_attribute(A, D, threshold, c, attr_cards):
    """ Returns the index of attr that leads to most info gain.
        If no a_i leads to gain >= threshold, returns None"""
    # need separate func if using gain ratio
    p0 = entropy(D, c)
    gain = None
    gain_i = 0
    for i in range(len(A)):
        if i != c and i != (c - 1):
            alpha = find_best_split(D, i, c)
            pAi = entropy_ai_alpha(D, i, alpha, c)
            temp_gain = p0 - pAi
            if gain is None or temp_gain > gain:
                gain = temp_gain
                gain_i = i
    if gain > threshold:
        return gain_i
    return None


def find_best_split(D, i, c):
    Gain = None
    max_gain_alpha = None
    p0 = entropy(D, c)
    potential_alphas = get_domain(D, i)
    sorted(potential_alphas)
    for j in potential_alphas:
        pAi = entropy_ai_alpha(D, i, j, c)
        temp_gain = p0 - pAi
        if (Gain is None) or (temp_gain > Gain):
            Gain = temp_gain
            max_gain_alpha = j
    return max_gain_alpha


def C45(D, A, threshold, c, attr_cards, entire_D):
    reverse_A = dict((y, x) for x, y in A.items())
    if classified(D, c) is not None:  # if D is completely uniform ()
        root = Node(D[0][c])
        root.p = 1.0
        return root
    elif not A:  # no more attributes to split on
        l, count = find_most_frequent_label_col(D, c)
        root = Node(l)
        root.p = count / len(D)
        return root
    else:
        # select splitting attribute
        Ag = select_splitting_attribute(A, D, threshold, c, attr_cards)
        if Ag is None:
            # max gain < threshold
            l, count = find_most_frequent_label_col(D, c)
            root = Node(l)
            root.p = count / len(D)
            return root
        else:
            # tree construction
            index_to = list(reverse_A.keys())[Ag]
            
            r = Node(reverse_A[index_to])
            start = time.time()
            alpha = find_best_split(D, Ag, c)
            D_plus, D_minus = split_on_alpha(D, Ag, alpha)
            minus_tree = C45(D_minus, A, threshold, c, attr_cards, entire_D)
            r.children += [("le", minus_tree)]
            plus_tree = C45(D_plus, A, threshold, c, attr_cards, entire_D)
            r.children += [("gt", plus_tree)]
            r.alpha = alpha

        return r

def classified(D, c):
    """ returns True if D[c] is same for all d in D, false otherwise"""
    if len(get_domain(D, c)) == 1:
        return get_domain(D,c)[0]    
    return None


def create_json_str(filename, root_node):
    json_str = f'{{"dataset":"{filename}",{create_node_str(root_node)}}}'
    return json_str


def create_edge_str(edge_label, child_node):
    json_str = f'{{"edge":{{"value":"{edge_label}",'
    json_str += f'{create_node_str(child_node)}'
    return json_str + "}}"

def create_num_edge_str(edge_label, child_node, alpha):
    json_str = f'{{"edge":{{"value":"{alpha}",'
    json_str += f'"direction":"{edge_label}",'
    json_str += f'{create_node_str(child_node)}'
    return json_str + "}}"


def create_node_str(node):
    if len(node.children) > 0:
        if node.alpha is not None:
            json_str = '"node":'
            json_str += f'{{"var":"{node.label}","edges":['
            for edge_label, child_node in node.children:
                json_str += f'{create_num_edge_str(edge_label, child_node, node.alpha)}' + ','

        else:
            json_str = '"node":'
            json_str += f'{{"var":"{node.label}","edges":['
            for edge_label, child_node in node.children:
                json_str += f'{create_edge_str(edge_label, child_node)}' + ','
        json_str = json_str[:-1]
        json_str += ']}'
    else:
        json_str = '"leaf":'
        json_str += f'{{"decision":"{node.label}",'
        json_str += f'"p":{node.p}}}'
    return json_str
