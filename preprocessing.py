from typing import overload
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.defchararray import count
from numpy.lib.function_base import average
import pandas as pd
import os
import sys
from num2words import num2words
from decimal import Decimal
from nltk.stem import PorterStemmer
import math
import time
import collections
import csv
import multiprocessing



class Vector():
    def __init__(self, index):
        self.index = index
        self.words = []
        self.tf_idf = []
        self.cosine_similarity = []
        self.okapi = []
        self.frequency = []

def get_stopwords_from_file(file):
        with open(file, 'r') as f:
            stopwords = f.read().splitlines()
            #get rid of spaces  
            stopwords = [word.strip() for word in stopwords]
            #lowercase
            stopwords = [word.lower() for word in stopwords]

        return stopwords

def get_stopwords(type):
    #get stopwords from file
    if type == "short":
        stopwords = get_stopwords_from_file("stopwords/stopwords-short.txt")
    elif type == "long":
        stopwords = get_stopwords_from_file("stopwords/stopwords-long.txt")
    elif type == "medium":
        stopwords = get_stopwords_from_file("stopwords/stopwords-medium.txt")
    elif type == "mysql":
        stopwords = get_stopwords_from_file("stopwords/stopwords-mysql.txt")
    elif type == "onix":
        stopwords = get_stopwords_from_file("stopwords/stopwords-onix.txt")
    else:
        stopwords = []
    return stopwords


def get_words_from_file(file, stopwords, stemmer):
    with open(file, 'r') as f:
        words = f.read()
        words = words.replace("."," ")
        words = words.replace("-"," ")
        words = words.translate(str.maketrans('', '', '!"#$%&\'()*+.,-/:;<=>?@[\\]^_`{|}~')).lower()
        words = words.split()

        words = [word for word in words if word not in stopwords]
        #words = [num2words(word) if word.isdigit() == True else word for word in words]
        words = [word for word in words if len(word) > 1]
        #stem words nltk
        if (stemmer == True):
            words = [PorterStemmer().stem(word) for word in words]
    return words

def remove_ds_store(dir):
    #remove .DS_Store
    dir = [dir for dir in dir if dir != ".DS_Store"]
    return dir

def write_groundtruth(groundtruth_overall):
    #write groundtruth_overall to file
    with open("groundtruth_overall.txt", 'w') as f:
        for line in groundtruth_overall:
            f.write(str(line) + "\n")


def get_all_words(stopword_type, stemmer):
    #get stopwords
    stopwords = get_stopwords(stopword_type)
    #get all directories in C50
    c50_dirs = os.listdir("C50")
    #remove .DS_Store   
    c50_dirs = remove_ds_store(c50_dirs)
    #get all directories in c50_dirs
    all_words = []
    vectors = []
    groundtruth_overall = []
    count = 0
    sum_length = 0
    for dir in c50_dirs:
        #get all directories in C50/dir
        c50_dirs_dir = os.listdir("C50/" + dir)
        #remove .DS_Store
        c50_dirs_dir = remove_ds_store(c50_dirs_dir)
        #get all files in C50/dir/dir
        for next_dir in c50_dirs_dir:
            c50_dirs_dir_files = os.listdir("C50/" + dir + "/" + next_dir)
            #remove .DS_Store
            c50_dirs_dir_files = remove_ds_store(c50_dirs_dir_files)
            #get all files in C50/dir/dir/dir
            for file in c50_dirs_dir_files:
                #get all words in file
                #groundtruth_overall.append([file, next_dir])

                vector = Vector(count)
                words = get_words_from_file("C50/" + dir + "/" + next_dir + "/" + file, stopwords, stemmer)

                vector.words = collections.Counter(words)
             
                #make words a dictionary and set to frequency
                # words_dict = {}
                # for word in words:
                #     words_dict[word] = 0
                # for word in words:
                #     words_dict[word] += 1
                


                vectors.append(vector)
                all_words.extend(words)

                #get length of file
                length = len(words)
                sum_length += length


                count += 1
    
    average_length = sum_length / count
    #get all unique words and sort them
    all_words = list(set(all_words))
    all_words.sort()
    #write_groundtruth(groundtruth_overall)
    return all_words, vectors, average_length

# def tf_idf(all_words, vectors):
#     #get total number of documents
#     total_documents = len(vectors)
#     overall_tf_idf = []
#     #get tf_idf for each word
#     for vector in vectors:
#         words_dict = dict.fromkeys(all_words, 0)
#         for word in vector.words:
#             #get tf
#             tf = vector.words[word] / sum(vector.words.values())
#             #get idf
#             idf = math.log(total_documents / len(vector.words))
#             #get tf_idf
#             tf_idf = tf * idf
#             #add to words_dict
#             words_dict[word] = tf_idf
#         x = np.asarray(list(words_dict.values()), dtype=np.float32)
#         vector.tf_idf = x
#         overall_tf_idf.append(x)

#     return vectors, overall_tf_idf
                

def tf_idf(all_words_df, vectors):
    #tf_idf
    #get total number of documents
    total_documents = len(vectors)
    overall_tf_idf = []
    #get tf_idf for each word
    for vector in vectors:
        words_dict = dict.fromkeys(all_words_df.keys(), 0)
        for word in vector.words:
            #get tf
            tf = vector.words[word] / sum(vector.words.values())
            #get idf
            idf = math.log(total_documents / all_words_df[word])
            #get tf_idf
            tf_idf = tf * idf
            #add to words_dict
            words_dict[word] = tf_idf
        x = np.asarray(list(words_dict.values()), dtype=np.float32)
        vector.tf_idf = x
        overall_tf_idf.append(x)
    return vectors, overall_tf_idf





    # #get total number of documents
    # total_documents = len(vectors)
    # overall_tf_idf = []
    # #get tf_idf for each word
    # count = 0
    # for vector in vectors:
    #     print(count)
    #     count +=1
    #     for i in range(len(vector.frequency)):
    #         #get tf
    #         tf = vector.frequency[i] / sum(vector.frequency)
    #         #get idf
    #         idf = math.log(total_documents / all_words_array[i])
    #         #get tf_idf
    #         tf_idf = tf * idf
    #         #add to words_dict
    #     x = np.asarray(tf_idf, dtype=np.float32)
    #     vector.tf_idf = x
    #     overall_tf_idf.append(x)
    #     return vectors, overall_tf_idf

    # return vectors, overall_tf_idf
                
            

def get_frequency_matrix(all_words, vectors):
    #make all_words a dictionary and set to 0
    y = dict.fromkeys(all_words, 0)
    overall_frequency = []
    for vector in vectors:
        x = vector.words
        res = {key: x.get(key, y[key]) for key in y}
        vector.frequency = np.asarray(list(res.values()), dtype=np.float32)
        overall_frequency.append(vector.frequency)
        # for (word, value) in x.items():
        #     x = all_words.index(word)
        #     words_array[x] += value
        # vector.frequency = words_array
        #vector.frequency = [words_array[all_words.index(word)] + value for word, value in x.items()]
        #vector.frequency = np.asarray([vector.words.count(word) for word in all_words], dtype=np.float32)
        #vector.frequency =  np.asarray([len(vector.words[vector.words == word]) for word in all_words], dtype=np.float32)
        #vector.frequency = sum([vector.words == x for x in all_words])
        # x = np.count_nonzero(vector.words == all_words)
        #vector.frequency = np.asarray([np.count_nonzero(vector.words == word) for word in all_words], dtype=np.float32)
    return overall_frequency, vectors

def cosine_similarity(vector1, overall_tf_idf, overall_tf_idf_magnitude):
    #get dot product of vector1 and overall_tf_idf
    dot_product = np.dot(overall_tf_idf, vector1)
    #get magnitude of vector1
    vector1_magnitude = np.linalg.norm(vector1)
    #get magnitude of overall_tf_idf
    #get cosine similarity
    cosine_similarity = dot_product / (vector1_magnitude * overall_tf_idf_magnitude)
    return cosine_similarity


def get_cosine_similarity(vectors, overall_tf_idf):
    #get cosine similarity for each vector
    overall_cosine_similarities = []
    overall_tf_idf_magnitude = np.linalg.norm(overall_tf_idf)
    for vector in vectors:
        x = cosine_similarity(vector.tf_idf, overall_tf_idf, overall_tf_idf_magnitude)
        overall_cosine_similarities.append(x)
        vector.cosine_similarity = x
    return overall_cosine_similarities, vectors
       
def get_okapi_similarity(vectors, all_words_frequency, average_length, overall_frequency_matrix):
    k1 = 1.5
    k2 = 500
    b = 0.75
    n = len(vectors)

    all_words_frequency = np.tile(all_words_frequency, (n, 1))

    x1 = np.log( (n - all_words_frequency + 0.5)/(all_words_frequency + 0.5) ) 

    x2 = ( (k1+1) * overall_frequency_matrix ) / (k1 * (1 - b + b * np.sum(overall_frequency_matrix)/average_length) + overall_frequency_matrix) 
  
    x = x1 * x2

    y = ( (k2+1) *  overall_frequency_matrix ) / (k2 +  overall_frequency_matrix )


    overall_okapi_similarities = []
    for i in range(len(y)):
        #timer start
        okapi_similarity = np.dot(x, y[i])
        #timer en
        overall_okapi_similarities.append(okapi_similarity)
    return overall_okapi_similarities, vectors

def get_df_frequency(all_words, vectors):
    #set np array of len all_words to 0
    y = dict.fromkeys(all_words, 0)
    for vector in vectors:
        x = vector.words
        res = {key: y.get(key, y[key]) + 1 for key in x}
        res2 = {key: res.get(key, y[key]) for key in y}
        y = res2
    return y 
    all_words_df = np.asarray(list(y.values()), dtype=np.float32)
    print(all_words_df)
    return all_words_df


def preprocessing(stopwords):
    stopword_type = stopwords[0]
    stemming = stopwords[1]

    #time
    start_time = time.time()
    all_words, vectors, average_length = get_all_words(stopword_type, stemming) #make sure to add in ability to change short to long stopwords
    #end time
    end_time = time.time()
    print("Time to get all words: " + str(end_time - start_time))


    #start time
    start_time = time.time()
    #get all words array
    all_words_frequency = get_df_frequency(all_words, vectors)
    #end time
    end_time = time.time()
    print("Time to get df frequency: " + str(end_time - start_time))


    #start time
    start_time = time.time()
    overall_frequency_matrix, vectors = get_frequency_matrix(all_words, vectors)
    #end time
    end_time = time.time()
    print("Time to get frequency matrix: " + str(end_time - start_time))


    # #start time
    # start_time = time.time()
    # vectors, overall_tf_idf = tf_idf(all_words_frequency, vectors)
    # #end time
    # end_time = time.time()
    # print("Time to get tf_idf: " + str(end_time - start_time))
    # #turn overall_tf_idf into numpy array
    # overall_tf_idf = np.asarray(overall_tf_idf, dtype=np.float32)
    

    # #start time
    # start_time = time.time()
    # #get cosine similarity
    # overall_cosine_similarities, vectors = get_cosine_similarity(vectors, overall_tf_idf)
    # overall_cosine_similarities = np.asarray(overall_cosine_similarities, dtype=np.float32)

    # #end time
    # end_time = time.time()
    # print("Time to get cosine similarity: " + str(end_time - start_time))
    # #write overall_cosine_similarities to csv file
    # pd.DataFrame(overall_cosine_similarities).to_csv("overall_cosine_similarities_" + stopword_type + "_" + str(stemming) + ".csv")

    #start time
    start_time = time.time()
    #all_words_frequency as np array
    all_words_frequency = np.asarray(list(all_words_frequency.values()), dtype=np.float32)
    overall_frequency_matrix = np.asarray(overall_frequency_matrix, dtype=np.float32)
    #get okapi
    okapi_similarities, vectors = get_okapi_similarity(vectors, all_words_frequency, average_length, overall_frequency_matrix)
    #okapi similarities as np array
    okapi_similarities = np.asarray(okapi_similarities, dtype=np.float32)
    #end time
    end_time = time.time()
    print("Time to get okapi similarity: " + str(end_time - start_time))
    #write okapi similarities to csv file
    pd.DataFrame(okapi_similarities).to_csv("okapi_similarities_" + stopword_type + "_" + str(stemming) + ".csv")



if __name__ == '__main__':
    stopword_type = ["short", "long", "medium", "mysql", "onix", "empty"]
    stemming = [True, False]

    #merge stopword_type and stemming into one list list of tuples list comprehension
    stopword_stemming = [(stopword_type[i], stemming[j]) for i in range(len(stopword_type)) for j in range(len(stemming))]
 
    
    #run preprocessing for each stopword_stemming using multiprocessing pool
    p = multiprocessing.Pool(processes=len(stopword_stemming))
    p.map(preprocessing, stopword_stemming)
    p.close()
    p.join()

    
    #[preprocessing(stopwords) for stopwords in stopword_stemming]

    # #time
    # start_time = time.time()
    # pool = multiprocessing.Pool()
    # pool.map(preprocessing, stopword_stemming)
    # pool.close()
    # pool.join()
    # #end time
    # end_time = time.time()
    # print("Time to preprocess: " + str(end_time - start_time))
