import pprint, json, pickle
from sklearn.cluster import KMeans, Birch
import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

graph = {}
nodes = []

# Features
id_count = {}
od_count = {}
idw_count = {}
odw_count = {}
lcc_count = {}

f = []


def read(x):
    if len(x.split(',')) == 15:
        sip = x.split(',')[3]
        dip = x.split(',')[6]

        if sip not in nodes:
            nodes.append(sip)
        if dip not in nodes:
            nodes.append(dip)

        byteSize = int(x.split(',')[12])/int(x.split(',')[11])
        srcpkts = int(x.split(',')[13])/byteSize
        dstpkts = int(x.split(',')[11]) - srcpkts

        if sip not in graph:
            graph[sip] = {}

        if dip not in graph[sip]:
            graph[sip][dip] = srcpkts
            graph[dip][sip] = dstpkts
        else:
            graph[sip][dip] += srcpkts
            graph[dip][sip] += dstpkts

        return sip


def ID(node):
    c = 0
    for n in nodes:
        if n != node:
            if n in graph:
                if node in graph[n]:
                    c += 1

    return c

def OD(node):
    c = 0
    for n in nodes:
        if n != node:
            if node in graph:
                if n in graph[node]:
                    c += 1
    return c

def IDW(node):
    c = 0
    for n in nodes:
        if n != node:
            if n in graph:
                if node in graph[n]:
                    c += graph[n][node]
    return c


def ODW(node):
    c = 0
    for n in nodes:
        if n != node:
            if node in graph:
                if n in graph[node]:
                    c += graph[node][n]
    return c

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

def train():
    with open('./model/data42.csv', 'r') as file:
        whole = file.readlines()

    for x in whole[1:]:
        read(x)

    for n in nodes:
        f.append([ ID(n), OD(n), IDW(n), ODW(n), LCC(n)])

    with open('features.json', 'w') as fv:
        json.dump(f, fv, indent=4)

    X = np.array(f)

    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

    pickle.dump(kmeans, open("kmeans.pkl", "wb"))

def model(val):
    kmeans = pickle.load(open("kmeans.pkl", 'rb'))
    return kmeans.predict([val])

def json2csv(data):
    # StartTime,Dur,Proto,SrcAddr,Sport,Dir,DstAddr,Dport,State,sTos,dTos,TotPkts,TotBytes,SrcBytes,Label
    return str(data['StartTime'] + ',' + str(data['Dur']) + ',' + data['Proto'] + ',' + data['SrcAddr'] + ',' +\
        str(data['Sport']) + ',' + data['Dir'] + ',' + data['DstAddr'] + ',' + str(data['Dport']) + ',' +\
        data['State'] + ',' + str(data['sTos']) + ',' + str(data['dTos']) + ',' + str(data['TotPkts']) + ',' + str(data['TotBytes']) + \
        ',' + str(data['SrcBytes']) + ',' + data['Label'])

def test(data):
    line = json2csv(data)
    n = read(line)
    return model([ ID(n), OD(n), IDW(n), ODW(n), LCC(n)])
