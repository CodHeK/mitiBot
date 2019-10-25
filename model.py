import numpy as np
import pprint, re, random, pickle, json, argparse
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, Birch, DBSCAN
from sklearn.linear_model import LogisticRegression
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

    dbscan = DBSCAN(eps=0.5, min_samples=5).fit(X)

    pickle.dump(dbscan, open("./saved/dbscan.pkl", "wb"))

    print("Done with PHASE 1 of Training!")


def train_p2():
    # PHASE 2 - SUPERVISED LEARNING

    with open("./saved/f.json", "r") as feat:
        sf = json.load(feat)


    dbscan = pickle.load(open("./saved/dbscan.pkl", "rb"))
    preds = dbscan.labels_

    X = []
    y = []

    for i, item in enumerate(sf):
        if preds[i] != 0:
            X.append(sf[item][0])
            y.append(sf[item][1])

    LR = LogisticRegression().fit(X, y)

    pickle.dump(LR, open("./saved/LR.pkl", "wb"))

    print("Done with PHASE 2 of Training! - DBSCAN")


def test():
    LR = pickle.load(open("./saved/LR.pkl", "rb"))

    with open("./saved/f.json", "r") as feat:
        sf = json.load(feat)

    acc = 0
    for i, item in enumerate(sf):
        if str(LR.predict([ sf[item][0] ])[0]) == str(sf[item][1]):
            acc += 1

    print("Accuracy: " + str((acc*100)/float(len(sf))) + " % - (DBSAN + LR)")



if __name__ == '__main__':
    # FLAGS
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", help="trains model", action="store_true")
    parser.add_argument("--phase1", help="trains model", action="store_true")
    parser.add_argument("--phase2", help="trains model", action="store_true")
    parser.add_argument("--test", help="evaluates model", action="store_true")

    args = parser.parse_args()

    start = datetime.now()
    start_time = start.strftime("%H:%M:%S")
    print("Start Time =", start_time)

    #########################################

    if args.train:
        b = Build('./datasets/42.csv')
        b.data = b.build_train_set(b.non_bot_tuples, b.bot_tuples)
        b.preprocess()

        print("Done pre-processing on Train set!")

        train_p1()

        train_p2()

    if args.phase1:
        # PRE-PROCESS THE TRAINING DATASET & UNSUPERVISED LEARNING
        b = Build('./datasets/42.csv')
        b.data = b.build_train_set(b.non_bot_tuples, b.bot_tuples)
        b.preprocess()

        print("Done pre-processing on Train set!")

        train_p1()

    if args.phase2:
        # PERFORM SUPERVISED LEARNING
        train_p2()

    if args.test:
        t = Build('./datasets/43.csv')
        t.data = t.build_test_set(t.non_bot_tuples, t.bot_tuples, 10)
        t.preprocess()

        print("Done pre-processing on Test set!")

        test()

    #########################################

    end = datetime.now()
    end_time = end.strftime("%H:%M:%S")
    print("End Time =", end_time)
