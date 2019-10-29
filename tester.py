import numpy as np
import pprint, re, random, pickle, json, argparse
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, Birch, DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def test():
    # LR = pickle.load(open("./saved/LR.pkl", "rb"))
    kmeans = pickle.load(open('kmeans.pkl', 'rb'))

    with open("./saved/f.json", "r") as feat:
        sf = json.load(feat)

    # bots = 0
    # for it in sf:
    #     if str(sf[it][1]) == '1':
    #         bots += 1
    #
    # print(bots)

    y_true = []
    y_pred = []
    acc = 0
    for i, item in enumerate(sf):
        if str(kmeans.predict([ sf[item][0] ])[0]) == str(sf[item][1]):
            acc += 1
        else:
            print(sf[item][0])

        y_true.append(str(sf[item][1]))
        y_pred.append(str(kmeans.predict([ sf[item][0] ])[0]))

    # yt = {}
    # for i in y_true:
    #     if i not in yt:
    #         yt[i] = 1
    #     else:
    #         yt[i] += 1
    #
    # yp = {}
    # for i in y_pred:
    #     if i not in yp:
    #         yp[i] = 1
    #     else:
    #         yp[i] += 1
    #
    # print(yt)
    # print(yp)

    print("Accuracy: " + str((acc*100)/float(len(sf))) + " %")

if __name__ == '__main__':
    test()
