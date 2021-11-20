import sys 
import pandas as pd 
import math 
import json 
import numpy as np
import random
from collections import Counter
from timeit import default_timer as timer

def prepare_D(training_file, restrictions_list): 
    data = pd.read_csv(training_file)
    D = pd.DataFrame(data) #See if you want to do a multi-hot encoding structure for ease of use 
    n = is_numeric(D)
    value = D.iloc[1][0]
    numbers_list = D.copy()
    numbers_list = list(numbers_list.iloc[0])
    D = D.iloc[2:]
    A = list(D)
    ignore_list = []
    ignore_list.append(value)

    for i in range(len(numbers_list)):
        if type(numbers_list[i]) != float: 
            if int(numbers_list[i]) == -1: 
                ignore_list.append(A[i])

        if len(restrictions_list) > 0: 
            if (int(restrictions_list[i]) == 0): 
                ignore_list.append(A[i])
        i+=1 

    D = D[[c for c in D if c not in ignore_list] + [value]]
    A = list(D)
    return n, D, A

def is_numeric(D): 
  numeric = {}
  for col in D.columns:
    if int(D.at[0,col]) == 0: 
      numeric[col] = True 
    else: 
      numeric[col] = False 

  return numeric

def find_result(attributes, list_, dict_, D): 
  result = ""
  if "node" in  dict_.keys():
    node = dict_["node"]
    result = find_result(attributes, list_,node, D)
  elif "leaf" in  dict_.keys():
    return dict_['leaf']['decision']
    
  elif "edges" in  dict_.keys():
    edges = dict_["edges"]

    first_edge = edges[0]['edge']
    if "direction" in first_edge:
      
      edge_value = first_edge["value"]
      second_edge = edges[1]['edge']
      attribute = dict_["var"]

      try:
        columns = D.columns.to_list()
        passed_in_value = float(list_[attributes.index(str(attribute))])
      except:
        print("Error")
     
        sys.exit()

      if passed_in_value <= edge_value:
        if first_edge["direction"] == "le": 
          result =  find_result(attributes, list_, first_edge["node"], D)
        else: 
          result =  find_result(attributes, list_,second_edge["node"], D)
          
      else: 
        if first_edge["direction"] == "gt": 
          result =  find_result(attributes, list_, first_edge["node"], D)
        else: 
          result =  find_result(attributes, list_,second_edge["node"], D)

    else: 
      for edge in edges: 
        if edge['edge']['value'] in list_: 
          edge_of_interest = edge['edge']
          if "leaf" in edge_of_interest.keys():
            return  edge['edge']['leaf']['decision']
          else: 
            result = find_result(attributes, list_,edge_of_interest["node"], D)
  else:
    result = np.nan
  
  return result 

def zero_matrix(result):
  zeros = []
  for i in range(len(result)):
    zeros.append([0]*len(result))

  actual = [] 
  classified = []
  for i in result:
    actual.append("A "+ str(i))
    classified.append("C "+ str(i))

  data = pd.DataFrame(data = zeros, index = actual, columns = classified)

  return data

def matrix(a,b, result, class_labels):
  data = zero_matrix(class_labels)
  x=0
  for i, j in zip(a, b):
    x = x + 1
    data["C "+ str(i)]["A "+ str(j)] = data["C "+ str(i) ]["A "+ str(j)] + 1
  return data

def classifier2(D, raw, class_labels):
  objects = list(D[D.columns[-1]])
  attributes = list(D.iloc[:1])

  overall = []
  overall_r = []
  o_data = []
  records = len(D)
  for index, row in D.iterrows():
    passed_row = row.tolist()
    if len(attributes) != len(passed_row): 
      print("ERROR")
      sys.exit()


    result = find_result(attributes, passed_row, raw, D) #C45 implimentation 
  

    if(result == ''):
      result = np.nan
    overall.append(result)

  x = D[D.columns[-1]].to_list()

  data = {'Predicted': overall, "Real": x}
  df = pd.DataFrame(data)

  df = df.dropna()

  a = df['Predicted'].to_list()
  b = df['Real'].to_list()

  result = df['Real'].value_counts().index.to_list()

  data = matrix(a,b, result, class_labels)
  o_data.append(data)
  overall_r.append(result)

  return records, data, result 

def cross_validation2(is_numeric, D, A, n, threshold):
  D = D.sample(frac = 1).reset_index(drop=True)
  overall = len(D)
  o_eval = len(D)
  correct = 0
  total_amt = 0
  accuracy = []

  if(n == 0):
    n = 1
  elif(n == -1):
    n = len(D)

  i = 1 
  result = []
  matrix_fin = []
  while i <= n:
    num = round(overall/n)
    if (num > o_eval or (i - n) == 0):
      num = o_eval
    o_eval = o_eval - num
    end = total_amt + num - 1
    if (total_amt+num) > overall:
      end = overall

    send_rec = D.loc[total_amt:end].reset_index(drop=True)
    

    if (n == 1):
      send_C45 = send_rec
    else:
      send_C45 = D.drop(range(total_amt,end)).reset_index(drop=True)
    

    before_node = Normal()
    dict_ = {}
    Attributes = A.copy()

    class_label = D[D.columns[-1]].value_counts().to_dict()
    
    node, final_dict = C45(is_numeric, send_C45, Attributes, threshold, before_node, dict_, class_label)  

    i+=1

    total, data, result = classifier2(send_rec, final_dict, class_label)

    
    matrix_fin = []
    lst = data.values.tolist()
    total = data.values.sum()

    T = 0
    for j in range(len(lst)):
      T = T + lst[j][j]


    correct += T
    accuracy.append(T/total)
    matrix_fin.append(data)
    total_amt += total


  overall_matrix = zero_matrix(class_label)
  for j in matrix_fin:
    overall_matrix = overall_matrix.add(j, fill_value=0)
  print(overall_matrix)

  print("Totals: ")
  print("Overall Accuracy: ", correct/total_amt)
  print("Individual accuracies: ", accuracy)
  print("Average accuracy: ", sum(accuracy)/len(accuracy))

def C45(is_numeric, D, A, threshold, node, dict_, class_labels):
  best = []

  bol, attr = check_home(D, A)

  if (bol == True): 
    leaf = Leaf(1,attr)
    if type(attr) == tuple: 
      a = attr[0]
    else: 
      a = attr 
    leaf_dict = {"leaf": {"decision": a, "p":1}}
    return leaf , leaf_dict   

  elif not A:
    c, p = find_most_frequent_label(D)
 
    leaf = Leaf(1,c)
    leaf_dict = {"leaf": {"decision": c, "p": p}}
    return leaf , leaf_dict 
    
  else:
    Ag, x = selectSplittingAttribute(is_numeric, A,D,threshold, class_labels)

    if Ag == None:
      c, p = find_most_frequent_label(D)
      leaf = Leaf(1,c)
      leaf_dict = {"leaf": {"decision": c, "p": p}}
      return leaf, leaf_dict 

    else:
      try:
        A.remove(Ag)
      except:
        node = Normal(Ag)
        node_dict = {"node": { "var": x,"edges": []}}
        option_labels = ['le', 'gt']
        for i in range(len(option_labels)):
          if option_labels[i] == 'le':
            data = D[D[x] <= Ag]
          else:
            data = D[D[x] > Ag]
        
          if (data.empty):
            # NEW GHOST PATHS CODE
            c, p = find_most_frequent_label(D)
            leaf = Leaf(1,c)
            leaf_dict = {"leaf": {"decision": c, "p": p}}
            return leaf, leaf_dict  

          child, child_dict = C45(is_numeric, data, A, threshold, node, dict_, class_labels)
          edge = Edge(option_labels[i], child)
          edge_dict = {"edge": {"value": float(Ag), "direction": option_labels[i], "node": child_dict}}
          node.edges.append(edge)
          node_dict["node"]['edges'].append(edge_dict)

      else:
        node = Normal(Ag)
        node_dict = {"node": { "var": Ag,"edges": []}}
        
        labels = list(D[Ag])
        option_labels = list(set(labels))

        for i in range(len(option_labels)):
          try:
            data = D.loc[(D[Ag] == option_labels[i])] 
          except:
            continue
          else:
            child, child_dict = C45(is_numeric, data, A, threshold, node, dict_, class_labels)
            
            edge = Edge(option_labels[i], child)

            edge_dict = {"edge": {"value": option_labels[i], "node": child_dict}}
            node.edges.append(edge)
            node_dict["node"]['edges'].append(edge_dict)
            

  return node, node_dict

class Leaf:
    def __init__(self,probability, result):
      self.probability = probability
      self.result = result
  
    def __repr__(self):
      ret = "{} {}".format( self.probability, self.result)
      return ret 

class Normal:
    def __init__(self, A=None):
      self.A = A
      self.edges = []
  
    def __repr__(self):
      ret1 = "{}".format( self.A)
      ret2 = ""
      for edge in self.edges: 
        ret2 += "{}".format( edge.__repr__())
      return ret1 + ret2  

class Edge:
    def __init__(self, label, node):
      self.label = label 
      self.node = node  
    def __repr__(self):
      ret = "{} {}".format( self.label,  self.node.__repr__())
      return ret 

def find_most_frequent_label(D): 
  df = D.apply(pd.Series.value_counts)
  max_column = df.columns[-1]
  label = df[max_column].idxmax()
  percentage = df[max_column].max()/df[max_column].sum()
  return label, percentage

def selectSplittingAttribute(is_numeric, A,D,threshold, class_labels):
  gain = []
  check = []

  p0, option_labels = first_entropy(D, D.columns[-1])

  
  pA = 0
  for i in range(len(A)):
    if is_numeric[A[i]] == True:
      x, pA = findBestSplit(D.columns[-1], A[i], D, class_labels) #returns best splitting attribute and calculated pA

    else: 
      pA = entropy(D, A[i], D.columns[-1])
      x = 'category'
      
    check.append(x)
    gain.append(p0 - pA) 
  
  best = max(gain)

  if best > threshold:
    mx = A[gain.index(best)]
    x = check[gain.index(best)]
    if x != 'category':
      return x, mx
    else:
      return mx, x
  else:
    return None, None
  
def findBestSplit(A, a, data, class_labels):
  n=len(data)

  values_2 = np.sort(data[a].unique()).astype(float)
  if (len(values_2) ==1):
    values_2 =  values_2
  else:
    values_2 = values_2[:-1]

  myAtt = a

  splits = np.array([data[data[myAtt] <= i] for i in values_2], dtype=object)

  sizes = np.array([len(splits[i]) for i in range(len(values_2))])

  distance = np.array([dict(data[data[a] <= i][A].value_counts()) for i in values_2])

  ffff = np.array([dict(Counter(dist) + Counter({x:1 for x in class_labels})) for dist in distance])

  counts = np.array([{x:fff[x]-1 for x in fff} for fff in ffff])

  class_l = dict(Counter(class_labels) + Counter({x:1 for x in class_labels}))

  rrrr = np.array([Counter(class_l) - Counter(i) for i in counts])
  rightSide = np.array([{x:rrr[x]-1 for x in rrr} for rrr in rrrr])
 
  counts = np.array([np.array(list(c.values())) for c in counts])
  rightSide = np.array([np.array(list(right.values())) for right in rightSide])
  
  f = np.array([c/l for c,l in zip(counts, sizes)])


  np.seterr(invalid='ignore')
  g = (rightSide-np.amin(rightSide))/(np.amax(rightSide)-np.amin(rightSide))

  split_entropies = (sizes/n * (-np.sum(f*np.log2(f, out=np.zeros_like(f), where=(f!=0)), axis = 1)))  +   ((n-sizes)/n * -np.sum(g*np.log2(g, out=np.zeros_like(g), where=(g!=0)), axis=1))

  return values_2[np.argmin(split_entropies)], np.min(split_entropies)

def first_entropy(D, A):
  labels = list(D[A])
  option_labels = list(set(labels))
  total = len(D)
  entropy = 0 
  for i in range (len(option_labels)):  
    first_ratio = int((D[A] == option_labels[i]).sum()) / total 
    entropy = entropy + log(first_ratio)
  return entropy, option_labels 

def entropy(D, a, A):
  entropy1, option_labels = first_entropy(D, A)
  if (a == A):
    return entropy1
  labels2 = list(D[a])
  option_labels2 = list(set(labels2))
  overall = 0
  for i in range (len(option_labels2)): 
    entropy = 0
    for j in range (len(option_labels)): 
      label = option_labels2[i]
      total = int((D[a] == label).sum())
      ratio_1 = len(D.loc[(D[a] == label) & (D[A] == option_labels[j]) ].index) / total
      entropy = entropy + log(ratio_1)
    overall = overall + (total/len(D[a])) * entropy
  return overall

def log(A):
  try:
    x = math.log2(A)
  except:
    x = 0
  return -((A) * x)

def check_home(D, A):
  d = D[D.columns[-1]]
  results = list(d.unique())
  if len(results) > 1: 
    return False, False 
  else: 
    return True, results[0]







