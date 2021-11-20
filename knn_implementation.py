import sys 
import pandas as pd 
import math 
import json 
import numpy as np
import random

from pandas.core.tools.numeric import to_numeric
from utils import *
import time
from contextlib import redirect_stdout
import multiprocessing


def prepare_D_knn(training_file, ground_truth_file): 
    D = pd.read_csv(training_file, sep=',', engine='c', na_filter=False, low_memory = False)
    D = D.drop(columns=['Unnamed: 0'])
    D = D.reset_index(drop=True)
    
    A = pd.read_csv(ground_truth_file, sep=',', engine='c', na_filter=False, low_memory = False)
    A = A.reset_index(drop=True)

    #merge D and A into one dataframe

    D = pd.merge(D, A, left_index=True, right_index=True)
    D = D.reset_index(drop=True)

    return D

def classifier_knn(D, k, is_numeric, class_labels):

    D_num = D.drop(columns=['author', 'file'])
    x = D[D.columns[-1]].to_list()
    D_num = np.asarray(D_num, dtype=np.float32)
    predicted = []
    for i in range(len(D)):
        dist = D_num[i]
        #put x and dist into a dictionary
        numbers = dict(zip(dist, x))
        #sort the dictionary by key lowest to highest
        predictions = sorted(numbers.items(), reverse=True)[1:k+1]
        # predictions = sorted(numbers.items(), key=lambda kv: kv[0])
        #get most repeated second element of the list of tuples
        predictions = [x[1] for x in predictions]
        #return most repeated element
        prediction = (max(set(predictions), key=predictions.count))
        predicted.append(prediction)
     

        # #timer
        # start = time.time()
        # df = pd.DataFrame({'Distance': dist, "Real": x})
        # df = df.sort_values(by = ['Distance']).reset_index(drop=True)
        # print(df['Real'].value_counts())
        # # print(df[1:k+1]['Real'].value_counts())
        # prediction = df[1:k+1]['Real'].value_counts().index[0]
        # end = time.time()
        # print("Time df taken: ", end - start)

        # predicted.append(prediction)
    data = {'Predicted': predicted, "Real": x}
    df = pd.DataFrame(data)

    result = df['Real'].value_counts().index.to_list()

    data = matrix(predicted, x, result, class_labels)

    records = len(D)
    
    return records, data, result 
 

def print_results(data, author):

    #find total of list of lists data using numpy and list comprehension
    total = sum([sum(x) for x in data])

    T = 0
    precision = []
    recall = []
    fmeasure = []
    hits = []
    strikes = []
    misses = []
    for i in range(len(data)):
        FN = 0
        FP = 0

        TP = data[i][i]
        T = T + data[i][i]
        hits.append(TP)
        for j in range(len(data)):
            if i != j:
                FN += data[i][j]
                FP += data[j][i]

        misses.append(FN)
        strikes.append(FP)
        TN = total - T - FN - FP

        if (TP + FP) == 0:
            p = 0
        else:
            p = TP / (TP + FP)
        precision.append(p)

        if (TP + FN) == 0:
            r = 0
        else:
            r = TP / (TP + FN)
        recall.append(r)

        if (p + r) == 0:
            f = 0
        else:
            f = 2 * (p * r) / (p + r)
        fmeasure.append(f)
    
    df = pd.DataFrame({"Author": author, "Hits": hits, "Misses": misses, "Strikes": strikes, "Precision": precision, "Recall": recall, "F-Measure": fmeasure})
    df = df.sort_values(by = ['F-Measure'], ascending=False)

    print(df)


    correct = T
    incorrect = total - T
    accuracy = T/total 
    print()
    print("Matrix Totals")
    print()
    #grab best author by f-measure
    print("Best Author: ", df.iloc[0]['Author'])
    print("Correct: ", correct)
    print("Incorrect: ", incorrect)
    print("Accuracy: ", accuracy)
    #matrix_fin.append(data)


def generate_knn(stopword_stemming):
    k = stopword_stemming[1]
   
    training_file = stopword_stemming[0] + "_" + stopword_stemming[2] + "_" + str(stopword_stemming[3]) + ".csv" #watch out 
    ground_truth_file = "groundtruth_overall.csv"

    
    D = prepare_D_knn(training_file,  ground_truth_file)

    #cross validation

    class_labels = D[D.columns[-1]].value_counts().to_dict()

    records, data, result = classifier_knn(D, k, is_numeric, class_labels)
    matrix_fin = []

    authors = D['author'].unique()
    lst = data.values.tolist()

    print()
    print("KNN Values: K = " + str(k))
    print()
    print_results(lst, authors)

def generate_outputs(stopword_stemming):
    with open(stopword_stemming[0] + "_" + stopword_stemming[2] + "_" + str(stopword_stemming[3]) + "_k=" + str(stopword_stemming[1]) + ".txt", 'w') as f:
        with redirect_stdout(f):
            generate_knn(stopword_stemming)

    
if __name__ == "__main__":
    words = ["overall_cosine_similarities", "okapi_similarities"]
    k_list = [1, 5, 9, 13, 15, 19, 23]
    stopword_type = ["short", "long", "medium", "mysql", "onix", "empty"]
    stemming = [True, False]


    #merge stopword_type and stemming into one list list of tuples list comprehension
    stopword_stemming = [(words[b], k_list[z], stopword_type[i], stemming[j]) for b in range(len(words)) for z in range(len(k_list)) for i in range(len(stopword_type)) for j in range(len(stemming))]

    p = multiprocessing.Pool(processes=16)
    p.map(generate_outputs, stopword_stemming)
    p.close()
    p.join()




    