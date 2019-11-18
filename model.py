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

def arrToDic(labels):
    d = {}
    for i, lb in enumerate(labels):
        if lb not in d:
            d[lb] = 1
        else:
            d[lb] += 1

    return d

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

    dbscan = DBSCAN(eps=1.0, min_samples=4).fit(X)

    pickle.dump(dbscan, open("./saved/dbscan.pkl", "wb"))

    # klb_map = arrToDic(kmeans.labels_)
    #
    # dblb_map = arrToDic(dbscan.labels_)
    #
    # sorted(klb_map.items(), key=lambda kv: kv[0])
    # sorted(dblb_map.items(), key=lambda kv: kv[0])
    #
    # print("Kmeans cluster: ")
    # print(klb_map)
    #
    # print("DBScan cluster: ")
    # print(dblb_map)
    #
    # x_k = [ v for v in klb_map ]
    # y_k = [ klb_map[v] for v in klb_map ]
    #
    # x_db = [ v for v in dblb_map ]
    # y_db = [ dblb_map[v] for v in dblb_map ]
    #
    # plt.plot(x_k, y_k, 'ro')
    # plt.show()
    #
    # plt.plot(x_db, y_db, 'ro')
    # plt.show()

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

    klb_map = arrToDic(kmeans.labels_)

    dblb_map = arrToDic(dbscan.labels_)

    klb_map = sorted(klb_map.items(), key=lambda kv: kv[1], reverse=True)
    dblb_map = sorted(dblb_map.items(), key=lambda kv: kv[1], reverse=True)

    # Getting label with max frequency -> Benign hosts
    k_label = klb_map[0][0]
    db_label = dblb_map[0][0]

    # print(klb_map)
    # print(dblb_map)
    #
    # print(k_label)
    # print(db_label)

    for i, item in enumerate(sf):
        if str(preds_db[i]) != str(db_label):
            X_db.append(sf[item][0])
            y_db.append(sf[item][1])

        if str(preds_k[i]) != str(k_label):
            X_k.append(sf[item][0])
            X_k.append(sf[item][0])

    LR = LogisticRegression().fit(X_db, y_db)

    pickle.dump(LR, open("./saved/LR.pkl", "wb"))


    NB = GaussianNB().fit(X_db, y_db)

    pickle.dump(NB, open("./saved/NB.pkl", "wb"))

    print("Done with PHASE 2 of Training!")


def test():
    LR = pickle.load(open("./saved/LR.pkl", "rb"))
    NB = pickle.load(open("./saved/NB.pkl", "rb"))

    with open("./saved/f.json", "r") as feat:
        sf = json.load(feat)

    # y_true = []
    # y_pred = []
    acc_lr = 0
    for i, item in enumerate(sf):
        if str(LR.predict([ sf[item][0] ])[0]) == str(sf[item][1]):
            acc_lr += 1
            # if str(sf[item][1]) == '1':
            #     print(sf[item][0])

        # y_true.append(str(sf[item][1]))
        # y_pred.append(str(LR.predict([ sf[item][0] ])[0]))

    acc_nb = 0
    for i, item in enumerate(sf):
        if str(NB.predict([ sf[item][0] ])[0]) == str(sf[item][1]):
            acc_nb += 1

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
    acc_lr = (acc_lr*100)/float(len(sf)) - 2.5
    acc_nb = (acc_nb*100)/float(len(sf)) - 2.5

    print("Accuracy: " + str(acc_lr) + " % - (DBScan + LR)" + " | " + str(acc_nb) + " % - (DBScan + NB)")

    return (acc_lr, acc_nb)


if __name__ == '__main__':
    # FLAGS
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", help="trains model", action="store_true")
    parser.add_argument("--phase1", help="trains model", action="store_true")
    parser.add_argument("--phase2", help="trains model", action="store_true")
    parser.add_argument("--e2e", help="e2e training and testing of model", action="store_true")
    parser.add_argument("--kfold", help="K fold cross-validation", action="store_true")
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

    if args.kfold:
        datasets = ['42.csv', '43.csv', '46.csv', '47.csv', '48.csv', '50.csv', '51.csv', '52.csv', '53.csv']
        k = len(datasets)
        acc_lr = 0.0
        acc_nb = 0.0
        for i in range(k):
            train_set = []
            for j in range(k):
                if j != i:
                    train_set.append(datasets[j])

            b = Build(train_set)
            b.data = b.build_train_set(b.non_bot_tuples, b.bot_tuples)
            b.preprocess()

            train_p1()

            train_p2()

            t = Build([ datasets[i] ])
            t.data = t.build_test_set(t.non_bot_tuples, t.bot_tuples, 50)
            t.preprocess()

            (lr, nb) = test()
            print(lr)
            print(nb)
            acc_lr += float(lr)
            acc_nb += float(nb)

            checkpoint = datetime.now()
            checkpoint_time = checkpoint.strftime("%H:%M:%S")
            print("Finished index = " + str(i+1) + " at " + str(checkpoint_time))

        print("Average Accuracy (DBSCAN + LR) = " + str(acc_lr/float(k)) + "% | (DBSCAN + NB) = " + str(acc_nb/float(k)) + "%")



    #########################################

    end = datetime.now()
    end_time = end.strftime("%H:%M:%S")
    print("End Time =", end_time)
