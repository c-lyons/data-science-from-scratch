import numpy as np
import pandas as pd
import random, string
import scipy.cluster.hierarchy as hac
import matplotlib.pyplot as plt
import scipy.spatial.distance as ssd


# creating random words frm which to modify and create clusters
def randomword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))

def randomletter():
    letters = string.ascii_lowercase
    return random.choice(letters)

def hamming_distance(a,b):
    if len(a) != len(b):
        raise RuntimeError('Hamming distance is valid for only equal length string')
    return sum(x!=y for x,y in zip(a,b))

def get_distances(i, j):
    return hamming_distance(i, j)

def generate_words(max_k, word_len):
    '''generates k random strings and returns as list of words'''
    # initialising X as empty list with up to K randomly generated words
    X = []
    max_k = 5
    n_clusters = (np.random.choice(range(max_k))+1)
    for k in range(n_clusters):
        X.append(randomword(word_len))
    print(f'There are {n_clusters} random words as initial seeds for the data...\n')
    print('The initial word(s) are: ', X)
    return X

def get_generations(n_iter, X, word_len, prob_mutate=0.05, num_child=3):
    '''Applies random mutations to input list of words and returns as full list'''
    print('Total Number of generations: ', n_iter)
    for i in range(n_iter):
        if i%500==0: print(f'Generation number {i}')
        for child in X:
            # for each child in X, adding mutation randomly to string
            children_list = []
            for children in range(num_child):
                rand = np.random.rand()
                if rand < prob_mutate:
                    new_word = list(child)  # converting to list temp to mutate string at index
                    # selecting rand index and replacing with rand letter
                    new_word[np.random.choice(word_len)] = randomletter()
                    children_list.append(''.join(new_word))  # replacing item at index n with new mutated word
                else:
                    children_list.append(child)
        X += children_list
    return X

def get_dist_matrix(X):
    '''gets distance matrix for distances between words in list of words'''
    print('Calculating distances...')
    dist_mat = np.zeros((len(X), len(X)))
    for n, i in enumerate(X):
        for m, j in enumerate(X):
            if n == m: pass  # if same index in list then pass (i.e. 0)
            else: dist_mat[n,m] = hamming_distance(i, j)
    print('Distance calculation finished.')
    return dist_mat

if __name__=='__main__':
    word_len = 10
    X = generate_words(5, word_len=word_len)
    X = get_generations(n_iter=1000, X=X, word_len=word_len, prob_mutate=0.03)
    dist_mat = get_dist_matrix(X)
    squareFormMat = ssd.squareform(dist_mat)

    tree = hac.linkage(squareFormMat, method="average", metric="euclidean")
    # plt.clf()
    plt.figure(figsize=(20,10))
    hac.dendrogram(tree)
    plt.show()