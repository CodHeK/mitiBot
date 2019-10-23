import numpy as np
import pprint, re, random, pickle, json, argparse
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, Birch, DBSCAN
from sklearn.linear_model import LogisticRegression


pp = pprint.PrettyPrinter(indent=4)


# GLOBAL VALUES
graph = {}
nodes = {}
node_map = {}

# FEATURES
id_count = {}
od_count = {}
idw_count = {}
odw_count = {}
lcc_count = {}

f = {}
fvecs = []

'''
    Flow ingestion
'''
def extract(self, x):
    if len(x.split(',')) == 15:
        sip = x.split(',')[3]
        dip = x.split(',')[6]

        nodes[sip] = 1

        byteSize = int(x.split(',')[12])/int(x.split(',')[11])
        srcpkts = int(x.split(',')[13])/byteSize
        dstpkts = int(x.split(',')[11]) - srcpkts
        flow = x.split(',')[14]

        if len(re.findall("Botnet", str(flow))) > 0:
            node_map[sip] = 1
        else:
            node_map[sip] = 0

        # SIP -> DIP
        if sip not in graph:
            graph[sip] = {}

        if dip not in graph[sip]:
            graph[sip][dip] = (srcpkts, (srcpkts, dstpkts))
        else:
            srcpktsPrev = graph[sip][dip][0]
            graph[sip][dip] = (srcpkts + srcpktsPrev, (srcpkts, dstpkts))

        if dip in graph:
            if sip in graph[dip]:
                srcpktsX = graph[sip][dip][0]
                dstpktsY = graph[dip][sip][1][1]
                graph[sip][dip] = (srcpktsX + dstpktsY, (srcpkts, dstpkts))


        # DIP -> SIP
        if dip not in graph:
            graph[dip] = {}

        if sip not in graph[dip]:
            graph[dip][sip] = (dstpkts, (srcpkts, dstpkts))
        else:
            dstpktsPrev = graph[dip][sip][0]
            graph[dip][sip] = (dstpkts + dstpktsPrev, (srcpkts, dstpkts))

        if sip in graph:
            if dip in graph[sip]:
                dstpktsX = graph[dip][sip][0]
                srcpktsY = graph[sip][dip][1][1]
                graph[dip][sip] = (dstpktsX + srcpktsY, (srcpkts, dstpkts))

        return (sip, dip)


'''
    Feature Extraction
'''

# IN-DEGREE (ID)
def ID(node):
    c = 0
    for n in nodes:
        if n != node:
            if n in graph:
                if node in graph[n]:
                    c += 1

    return c

# OUT-DEGREE (OD)
def OD(node):
    c = 0
    for n in nodes:
        if n != node:
            if node in graph:
                if n in graph[node]:
                    c += 1
    return c

# IN-DEGREE WEIGHT (IDW)
def IDW(node):
    c = 0
    for n in nodes:
        if n != node:
            if n in graph:
                if node in graph[n]:
                    c += graph[n][node][0]
    return c

# OUT-DEGREE WIGHT (ODW)
def ODW(node):
    c = 0
    for n in nodes:
        if n != node:
            if node in graph:
                if n in graph[node]:
                    c += graph[node][n][0]
    return c

# LOCAL CLUSTERING COEFFICIENT (LCC)
def LCC(node):
    N = 0
    NgNodes = []
    for n in nodes:
        if n != node:
            if node in graph:
                if n in graph[node]:
                    N += 1
                    NgNodes.append(n)

    F = 0
    for j in NgNodes:
        for k in NgNodes:
            if j != k:
                if j in graph:
                    if k in graph[j]:
                        F += 1

    if N > 1:
        return F/float(N*(N-1))
    else:
        return 0.0


'''
    F-Norm Implementation using D = 1
'''
def normalize(fn, node):
    N = 0
    sumF = np.array([0, 0, 0, 0, 0]).astype('float64')
    for n in nodes:
        if n != node:
            if node in graph:
                if n in graph[node]:
                    N += 1
                    sumF += np.array(f[n][0]).astype('float64')

    if N > 0:
        u = sumF/float(N)

        for idx, ui in enumerate(u):
            if ui == 0.0:
                ui += 0.01
            fn[idx] = fn[idx]/float(ui)

    return fn


'''
    Graph Transform and extracted feature storage
'''
def preprocess(content):
    for line in content:
        extract(line)

    print("Graph built!")

    # FOR EACH NODE CALCULATE THE FEATURE TUPLE [ F0, F1, F2, F3, F4 ]

    print(len(nodes))
    for node in nodes:
        f[node] = [ [ ID(node), OD(node), IDW(node), ODW(node), LCC(node) ], node_map[node] ]

    print("Feature Extraction done!")

    # NORMALIZE

    for node in nodes:
        f[node][0] = normalize(f[node][0], node)
        fvecs.append(f[node][0])

    with open('./saved/fvecs.json', 'w') as fv:
        json.dump(fvecs, fv, indent=4)

    with open('./saved/f.json', 'w') as feat:
        json.dump(f, feat, indent=4)

    print("Normalizing Done!")


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

    print("Done with PHASE 2 of Training!")


def test():
    pass


def mod(content):
    non_bot_tuples = []
    bot_tuples = []

    non_bot_flow_tuples = 97850

    for line in content:
        flow = line.split(',')[14]

        # NON BOTNET FLOWS
        if len(re.findall("Botnet", str(flow))) == 0:
            if non_bot_flow_tuples > 0:
                non_bot_flow_tuples -= 1
                non_bot_tuples.append(line)
        else:
            bot_tuples.append(line)

    random.shuffle(non_bot_tuples)
    random.shuffle(bot_tuples)

    test = non_bot_tuples[:10000] + bot_tuples[:5000]
    train = non_bot_tuples[10000:] + bot_tuples[5000:]

    random.shuffle(test)
    random.shuffle(test)

    return (train, test)


def read(filename):
    with open(filename, 'r') as file:
        content = file.readlines()
    file.close()
    return content


if __name__ == '__main__':
    # FLAGS
    parser = argparse.ArgumentParser()

    parser.add_argument("--phase1", help="trains model", action="store_true")
    parser.add_argument("--phase2", help="trains model", action="store_true")
    parser.add_argument("--test", help="evaluates model", action="store_true")

    args = parser.parse_args()

    start = datetime.now()
    start_time = start.strftime("%H:%M:%S")
    print("Start Time =", start_time)

    #########################################

    content = read('./datasets/42.csv')

    print("Read Dataset...")

    Train, Test = mod(content[1:])

    if args.phase1:
        # PRE-PROCESS THE TRAINING DATASET & UNSUPERVISED LEARNING
        preprocess(Train)

        print("Done pre-processing!")

        train_p1()

    if args.phase2:
        # PERFORM SUPERVISED LEARNING

        train_p2()

    if args.test:
        test()

    #########################################

    end = datetime.now()
    end_time = end.strftime("%H:%M:%S")
    print("Start Time =", end_time)
