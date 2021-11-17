import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
from num2words import num2words

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
    return stopwords
    
def get_words_from_file(file, stopwords):
    overall_words = []
    with open(file, 'r') as f:
        line = f.read().splitlines()
        for words in line:
            words = words.split()
            #get rid of spaces
            words = [word.strip() for word in words]
            #lowercase
            words = [word.lower() for word in words]
            #remove all punctuation
            words = [word.translate(str.maketrans('', '', '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')) for word in words]
            #get rid of stop words 
            words = [word for word in words if word not in stopwords]
            #get rid of words with only one letter
            words = [word for word in words if len(word) > 1]
            #number to word conversion
            words = [word.translate(str.maketrans('0123456789', ' '*10)) for word in words]
            
            #add words to overall_words
            overall_words.extend(words)
    return overall_words

def remove_ds_store(dir):
    #remove .DS_Store
    dir = [dir for dir in dir if dir != ".DS_Store"]
    return dir

def write_groundtruth(groundtruth_overall):
    #write groundtruth_overall to file
    with open("groundtruth_overall.txt", 'w') as f:
        for line in groundtruth_overall:
            f.write(str(line) + "\n")


def get_all_words(stopword_type):
    #get stopwords
    stopwords = get_stopwords(stopword_type)
    #get all directories in C50
    c50_dirs = os.listdir("C50")
    #remove .DS_Store   
    c50_dirs = remove_ds_store(c50_dirs)
    #get all directories in c50_dirs
    all_words = []
    groundtruth_overall = []
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
                groundtruth_overall.append([file, next_dir])
                words = get_words_from_file("C50/" + dir + "/" + next_dir + "/" + file, stopwords)
                all_words.extend(words)
    #get all unique words and sort them
    all_words = list(set(all_words))
    all_words.sort()
    #write_groundtruth(groundtruth_overall)
    return all_words

def tf_idf(all_words, stopword_type):
    #get stopwords
    stopwords = get_stopwords(stopword_type)
    #get all directories in C50
    c50_dirs = os.listdir("C50")
    #remove .DS_Store
    c50_dirs = remove_ds_store(c50_dirs)
    #get all directories in c50_dirs
    all_words = []
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
                pass

def frequency_matrix(all_words, stopword_type):
    #get stopwords
    stopwords = get_stopwords(stopword_type)
    #get all directories in C50
    c50_dirs = os.listdir("C50")
    #remove .DS_Store
    c50_dirs = remove_ds_store(c50_dirs)
    #get all directories in c50_dirs
    frequency_matrix = []
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
                words = get_words_from_file("C50/" + dir + "/" + next_dir + "/" + file, stopwords)
                #get frequency of all words in file in numpy array as type float
                print([words.count(word) for word in all_words])
                sys.exit()
                frequency_matrix.append(np.asarray([words.count(word) for word in all_words], dtype=np.float64))

    return frequency_matrix

def main():
    stopword_type = "short"
    all_words = get_all_words("short") #make sure to add in ability to change short to long stopwords
    print(len(all_words))
    print(all_words[:1000])
    frequency_matrix(all_words, stopword_type)

main()