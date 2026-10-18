from math import *
from decimal import Decimal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
import gower


class Similarity():
    """ Five similarity measures function """

    def euclidean_distance(self, x, y):
        """ return euclidean distance between two lists """

        return sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))

    def manhattan_distance(self, x, y):
        """ return manhattan distance between two lists """

        return sum(abs(a - b) for a, b in zip(x, y))

    def minkowski_distance(self, x, y, p_value):
        """ return minkowski distance between two lists """

        return self.nth_root(sum(pow(abs(a - b), p_value) for a, b in zip(x, y)),
                             p_value)

    def nth_root(self, value, n_root):
        """ returns the n_root of an value """

        root_value = 1 / float(n_root)
        return round(Decimal(value) ** Decimal(root_value), 3)

    def cosine_similarity(self, x, y):
        """ return cosine similarity between two lists """

        numerator = sum(a * b for a, b in zip(x, y))
        denominator = self.square_rooted(x) * self.square_rooted(y)
        return round(numerator / float(denominator), 3)

    def square_rooted(self, x):
        """ return 3 rounded square rooted value """

        return round(sqrt(sum([a * a for a in x])), 3)

    def jaccard_similarity(self, x, y):
        """ returns the jaccard similarity between two lists """

        intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
        union_cardinality = len(set.union(*[set(x), set(y)]))
        return intersection_cardinality / float(union_cardinality)


def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection

    return float(intersection)/union


def manhattan(a, b):
    return sum(abs(val1-val2) for val1, val2 in zip(a,b))


def nmf(data, labels):
    model = NMF(n_components=3, init='random', random_state=2)
    W = model.fit_transform(data)

    xs = W[:,0]
    ys = W[:,1]
    zs = W[:,2]

    plt.scatter(xs, ys, zs, alpha=0.5)
    for x,y,z, label in zip(xs,ys,zs,labels):
        plt.annotate(label, (x,y,z), fontsize=10, alpha=0.5)
    plt.show()


def gowersDist(data, labels):
    gowerMatrix = gower.gower_matrix(data)
    print(labels)
    print(gowerMatrix)
    gowerDists = {}
    for row in range(len(gowerMatrix)):
        gowerDists[labels[row]] = {}
        for col in range(len(gowerMatrix[row])):
            gowerDists[labels[row]][labels[col]] = gowerMatrix[row][col]

    for i in gowerDists:
        print(i)
        print(dict(sorted(gowerDists[i].items(), key=lambda item: item[1])))





