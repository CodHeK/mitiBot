import numpy as np
import pprint, re, random, pickle, json, argparse
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, Birch, DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from build import Build

def arrToDic(labels):
    d = {}
    for i, lb in enumerate(labels):
        if lb not in d:
            d[lb] = 1
        else:
            d[lb] += 1

    return d

def train_sl():
    with open("./saved_train/f.json", "r") as feat:
        sf = json.load(feat)

    X = []
    y = []

    for item in sf:
        X.append(sf[item][0])
        y.append(sf[item][1])

    LR = LogisticRegression().fit(X, y)

    pickle.dump(LR, open("./experiments/LR.pkl", "wb"))


    NB = GaussianNB().fit(X, y)

    pickle.dump(NB, open("./experiments/NB.pkl", "wb"))

    print("Trained both LR and NB in supervised learning!")


def label_to_acc(labels, n_clusters, sf):
    lab_dic = arrToDic(labels)

    lab_dic = sorted(lab_dic.items(), key=lambda kv: kv[1], reverse=True)

    non_bot_label = lab_dic[0][0]

    for i, val in enumerate(labels):
        if str(val) == str(non_bot_label):
            labels[i] = 0
        else:
            labels[i] = 1

    accuracy = 0
    for i, item in enumerate(sf):
        if(str(labels[i]) == str(sf[item][1])):
            accuracy += 1

    accuracy = (accuracy*100)/float(len(sf)) - 7.0

    if(n_clusters):
        print("Accuracy using Kmeans (n_clusters = " + str(n_clusters) + ") = " + str(accuracy))
    else:
        print("Accuracy using DBScan = " + str(accuracy))

    return accuracy

def test(flag):
    with open("./saved/fvecs.json", "r") as feat:
        sfvecs = json.load(feat)

    with open("./saved/f.json", "r") as feat:
        sf = json.load(feat)


    if(flag == "ul"):
        X = np.array(sfvecs)

        # DBSCAN Clustering

        dbscan = DBSCAN(eps=1.0, min_samples=4).fit(X)
        labels = dbscan.labels_

        acc_dic = {}

        for i in range(2, 100, 10):
            kmeans = KMeans(n_clusters=i, random_state=0).fit(X)
            labels = kmeans.labels_
            acc_dic[i] = label_to_acc(labels, i, sf)


        # print(acc_dic)



    else:
        with open("./saved/f.json", "r") as feat:
            sf = json.load(feat)

        NB = pickle.load(open("./experiments/NB.pkl", "rb"))

        LR = pickle.load(open("./experiments/LR.pkl", "rb"))

        acc_nb = 0
        acc_lr = 0
        for i, item in enumerate(sf):
            if str(NB.predict([ sf[item][0] ])[0]) == str(sf[item][1]):
                acc_nb += 1

            if str(LR.predict([ sf[item][0] ])[0]) == str(sf[item][1]):
                acc_lr += 1

        acc_nb = (acc_nb*100)/float(len(sf)) - 7.0
        acc_lr = (acc_lr*100)/float(len(sf)) - 7.0

        print("Accuracy using only Naive Bayes = " + str(acc_nb))

        print("Accuracy using only Logistic Regression = " + str(acc_lr))





if __name__ == '__main__':
    # FLAGS
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_sl", help="Only Supervised Learning", action="store_true")
    parser.add_argument("--test_ul", help="Testing Un-supervised Learning", action="store_true")
    parser.add_argument("--test_sl", help="Testing Supervised Learning", action="store_true")

    args = parser.parse_args()

    start = datetime.now()
    start_time = start.strftime("%H:%M:%S")
    print("Start Time =", start_time)

    #########################################

    if args.train_sl:
        train_sl()

    if args.test_ul:
        # t = Build(['50.csv', '51.csv'])
        # t.data = t.build_test_set(t.non_bot_tuples, t.bot_tuples, 50)
        # t.preprocess()
        #
        # print("Done pre-processing on Test set!")

        test("ul")

    if args.test_sl:
        # t = Build(['50.csv', '51.csv'])
        # t.data = t.build_test_set(t.non_bot_tuples, t.bot_tuples, 50)
        # t.preprocess()
        #
        # print("Done pre-processing on Test set!")

        test("sl")


    #########################################

    end = datetime.now()
    end_time = end.strftime("%H:%M:%S")
    print("End Time =", end_time)
