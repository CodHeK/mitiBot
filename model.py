import numpy as np
import pprint, re, random, pickle, json, argparse
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, Birch, DBSCAN
from sklearn.linear_model import LogisticRegression


pp = pprint.PrettyPrinter(indent=4)


class Build:
    def __init__(self, data):
        self.data = data
        self.graph = {}
        self.nodes = {}
        self.node_map = {}
        self.id_count = {}
        self.od_count = {}
        self.idw_count = {}
        self.odw_count = {}
        self.lcc_count = {}
        self.f = {}
        self.fvecs = []


    '''
        Flow ingestion
    '''
    def extract(self, x):
        if len(x.split(',')) == 15:
            sip = x.split(',')[3]
            dip = x.split(',')[6]

            self.nodes[sip] = 1

            byteSize = int(x.split(',')[12])/int(x.split(',')[11])
            srcpkts = int(x.split(',')[13])/byteSize
            dstpkts = int(x.split(',')[11]) - srcpkts
            flow = x.split(',')[14]

            if len(re.findall("Botnet", str(flow))) > 0:
                self.node_map[sip] = 1
            else:
                self.node_map[sip] = 0

            # SIP -> DIP
            if sip not in self.graph:
                self.graph[sip] = {}

            if dip not in self.graph[sip]:
                self.graph[sip][dip] = (srcpkts, (srcpkts, dstpkts))
            else:
                srcpktsPrev = self.graph[sip][dip][0]
                self.graph[sip][dip] = (srcpkts + srcpktsPrev, (srcpkts, dstpkts))

            if dip in self.graph:
                if sip in self.graph[dip]:
                    srcpktsX = self.graph[sip][dip][0]
                    dstpktsY = self.graph[dip][sip][1][1]
                    self.graph[sip][dip] = (srcpktsX + dstpktsY, (srcpkts, dstpkts))


            # DIP -> SIP
            if dip not in self.graph:
                self.graph[dip] = {}

            if sip not in self.graph[dip]:
                self.graph[dip][sip] = (dstpkts, (srcpkts, dstpkts))
            else:
                dstpktsPrev = self.graph[dip][sip][0]
                self.graph[dip][sip] = (dstpkts + dstpktsPrev, (srcpkts, dstpkts))

            if sip in self.graph:
                if dip in self.graph[sip]:
                    dstpktsX = self.graph[dip][sip][0]
                    srcpktsY = self.graph[sip][dip][1][1]
                    self.graph[dip][sip] = (dstpktsX + srcpktsY, (srcpkts, dstpkts))

            return (sip, dip)


    '''
        F-Norm Implementation using D = 1
    '''
    def normalize(self, fn, node):
        N = 0
        sumF = np.array([0, 0, 0, 0, 0]).astype('float64')
        for n in self.nodes:
            if n != node:
                if node in self.graph:
                    if n in self.graph[node]:
                        N += 1
                        sumF += np.array(self.f[n][0]).astype('float64')

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
    def preprocess(self):
        for line in self.data:
            self.extract(line)

        print("Graph built!")

        # FOR EACH NODE CALCULATE THE FEATURE TUPLE [ F0, F1, F2, F3, F4 ]

        print(len(self.nodes))
        for node in self.nodes:
            self.f[node] = [ [ self.ID(node), self.OD(node), self.IDW(node), self.ODW(node), self.LCC(node) ], self.node_map[node] ]

        print("Feature Extraction done!")

        # NORMALIZE

        for node in self.nodes:
            self.f[node][0] = self.normalize(self.f[node][0], node)
            self.fvecs.append(self.f[node][0])

        with open('./saved/fvecs.json', 'w') as fv:
            json.dump(self.fvecs, fv, indent=4)

        with open('./saved/f.json', 'w') as feat:
            json.dump(self.f, feat, indent=4)

        print("Normalizing Done!")


    '''
        Feature Extraction
    '''
    # IN-DEGREE (ID)
    def ID(self, node):
        c = 0
        for n in self.nodes:
            if n != node:
                if n in self.graph:
                    if node in self.graph[n]:
                        c += 1

        return c

    # OUT-DEGREE (OD)
    def OD(self, node):
        c = 0
        for n in self.nodes:
            if n != node:
                if node in self.graph:
                    if n in self.graph[node]:
                        c += 1
        return c

    # IN-DEGREE WEIGHT (IDW)
    def IDW(self, node):
        c = 0
        for n in self.nodes:
            if n != node:
                if n in self.graph:
                    if node in self.graph[n]:
                        c += self.graph[n][node][0]
        return c

    # OUT-DEGREE WIGHT (ODW)
    def ODW(self, node):
        c = 0
        for n in self.nodes:
            if n != node:
                if node in self.graph:
                    if n in self.graph[node]:
                        c += self.graph[node][n][0]
        return c

    # LOCAL CLUSTERING COEFFICIENT (LCC)
    def LCC(self, node):
        N = 0
        NgNodes = []
        for n in self.nodes:
            if n != node:
                if node in self.graph:
                    if n in self.graph[node]:
                        N += 1
                        NgNodes.append(n)

        F = 0
        for j in NgNodes:
            for k in NgNodes:
                if j != k:
                    if j in self.graph:
                        if k in self.graph[j]:
                            F += 1

        if N > 1:
            return F/float(N*(N-1))
        else:
            return 0.0



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
    LR = pickle.load(open("./saved/LR.pkl", "rb"))

    with open("./saved/f.json", "r") as feat:
        sf = json.load(feat)

    acc = 0
    for i, item in enumerate(sf):
        if str(LR.predict([ sf[item][0] ])[0]) == str(sf[item][1]):
            acc += 1


    print("Accuracy: " + str((acc*100)/float(len(sf))) + " %")


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

    parser.add_argument("--train", help="trains model", action="store_true")
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

    if args.train:
        b = Build(Train)
        b.preprocess()

        print("Done pre-processing on Train set!")

        train_p1()

        train_p2()

    if args.phase1:
        # PRE-PROCESS THE TRAINING DATASET & UNSUPERVISED LEARNING
        b = Build(Train)
        b.preprocess()

        print("Done pre-processing on Train set!")

        train_p1()

    if args.phase2:
        # PERFORM SUPERVISED LEARNING

        train_p2()

    if args.test:
        t = Build(Test)
        t.preprocess()

        print("Done pre-processing on Test set!")

        test()

    #########################################

    end = datetime.now()
    end_time = end.strftime("%H:%M:%S")
    print("Start Time =", end_time)
