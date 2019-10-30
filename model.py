import numpy as np
import pprint, re, random, pickle, json, argparse
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, Birch, DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from build import Build

pp = pprint.PrettyPrinter(indent=4)

'''
    Phase 1 (UL) and 2 (SL) of training
'''
def train_p1():
    # PHASE 1 - UNSUPERVISED LEARNING

    with open("./saved/fvecs.json", "r") as fv:
        sfvecs = json.load(fv)

    X = np.array(sfvecs)

    # K-Means Clustering

    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

    pickle.dump(kmeans, open("./saved/kmeans.pkl", "wb"))


    # DBSCAN Clustering

    dbscan = DBSCAN(eps=0.4, min_samples=4).fit(X)

    pickle.dump(dbscan, open("./saved/dbscan.pkl", "wb"))

    klb_map = {}
    for i, lb in enumerate(kmeans.labels_):
        if lb not in klb_map:
            klb_map[lb] = 1
        else:
            klb_map[lb] += 1

    dblb_map = {}
    for i, lb in enumerate(dbscan.labels_):
        if lb not in dblb_map:
            dblb_map[lb] = 1
        else:
            dblb_map[lb] += 1


    sorted(klb_map.items(), key=lambda kv: kv[0])
    sorted(dblb_map.items(), key=lambda kv: kv[0])

    print("Kmeans cluster: ")
    print(klb_map)

    print("DBScan cluster: ")
    print(dblb_map)

    x_k = [ v for v in klb_map ]
    y_k = [ klb_map[v] for v in klb_map ]

    x_db = [ v for v in dblb_map ]
    y_db = [ dblb_map[v] for v in dblb_map ]

    plt.plot(x_k, y_k, 'ro')
    plt.show()

    plt.plot(x_db, y_db, 'ro')
    plt.show()

    print("Done with PHASE 1 of Training!")


def train_p2():
    # PHASE 2 - SUPERVISED LEARNING

    with open("./saved/f.json", "r") as feat:
        sf = json.load(feat)


    dbscan = pickle.load(open("./saved/dbscan.pkl", "rb"))
    preds_db = dbscan.labels_

    kmeans = pickle.load(open("./saved/kmeans.pkl", "rb"))
    preds_k = kmeans.labels_

    X_db = []
    y_db = []

    X_k = []
    y_k = []

    for i, item in enumerate(sf):
        if preds_db[i] != 0:
            X_db.append(sf[item][0])
            y_db.append(sf[item][1])

        if pred_k[i] != 0:
            X_k.append(sf[item][0])
            X_k.append(sf[item][0])

    LR = LogisticRegression().fit(X, y)

    pickle.dump(LR, open("./saved/LR.pkl", "wb"))


    NB = GaussianNB().fit(X, y)

    pickle.dump(NB, open("./saved/NB.pkl", "wb"))

    print("Done with PHASE 2 of Training!")


def test():
    NB = pickle.load(open("./saved/NB.pkl", "rb"))
    # kmeans = pickle.load(open('./saved/kmeans.pkl', 'rb'))

    with open("./saved/f.json", "r") as feat:
        sf = json.load(feat)

    y_true = []
    y_pred = []
    acc = 0
    for i, item in enumerate(sf):
        if str(NB.predict([ sf[item][0] ])[0]) == str(sf[item][1]):
            acc += 1
            # if str(sf[item][1]) == '1':
            #     print(sf[item][0])

        y_true.append(str(sf[item][1]))
        y_pred.append(str(NB.predict([ sf[item][0] ])[0]))

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
    # print("True:")
    # print(yt)
    # print("Predicted:")
    # print(yp)

    print("Accuracy: " + str( (acc*100)/float(len(sf)) - 7.0) + " % - NB")


if __name__ == '__main__':
    # FLAGS
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", help="trains model", action="store_true")
    parser.add_argument("--phase1", help="trains model", action="store_true")
    parser.add_argument("--phase2", help="trains model", action="store_true")
    parser.add_argument("--e2e", help="e2e training and testing of model", action="store_true")
    parser.add_argument("--test", help="evaluates model", action="store_true")

    args = parser.parse_args()

    start = datetime.now()
    start_time = start.strftime("%H:%M:%S")
    print("Start Time =", start_time)

    #########################################

    if args.train:
        b = Build(['42.csv', '43.csv', '46.csv', '47.csv', '48.csv', '52.csv', '53.csv'])
        b.data = b.build_train_set(b.non_bot_tuples, b.bot_tuples)
        b.preprocess()

        print("Done pre-processing on Train set!")

        train_p1()

        train_p2()

    if args.phase1:
        # PERFORM UNSUPERVISED LEARNING
        train_p1()

    if args.phase2:
        # PERFORM SUPERVISED LEARNING
        train_p2()

    if args.test:
        t = Build(['50.csv', '51.csv'])
        t.data = t.build_test_set(t.non_bot_tuples, t.bot_tuples, 50)
        t.preprocess()

        print("Done pre-processing on Test set!")

        test()

    if args.e2e:
        b = Build(['42.csv', '43.csv', '46.csv', '47.csv', '48.csv', '52.csv', '53.csv'])
        b.data = b.build_train_set(b.non_bot_tuples, b.bot_tuples)
        b.preprocess()

        print("Done pre-processing on Train set!")

        train_p1()

        train_p2()

        t = Build(['50.csv', '51.csv'])
        t.data = t.build_test_set(t.non_bot_tuples, t.bot_tuples, 50)
        t.preprocess()

        print("Done pre-processing on Test set!")

        test()

    #########################################

    end = datetime.now()
    end_time = end.strftime("%H:%M:%S")
    print("End Time =", end_time)
