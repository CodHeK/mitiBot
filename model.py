import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)

# GLOBAL VALUES
graph = {}
nodes = {}

# FEATURES
id_count = {}
od_count = {}
idw_count = {}
odw_count = {}
lcc_count = {}

f = {}

def read(filename):
    with open(filename, 'r') as file:
        content = file.readlines()
    file.close()
    return content

def extract(x):
    if len(x.split(',')) == 15:
        sip = x.split(',')[3]
        dip = x.split(',')[6]

        if sip not in nodes:
            nodes[sip] = 1
        if dip not in nodes:
            nodes[dip] = 1

        byteSize = int(x.split(',')[12])/int(x.split(',')[11])
        srcpkts = int(x.split(',')[13])/byteSize
        dstpkts = int(x.split(',')[11]) - srcpkts

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


def normalize(fn, node):
    N = 0
    sumF = np.array([0, 0, 0, 0, 0]).astype('float64')
    for n in nodes:
        if n != node:
            if node in graph:
                if n in graph[node]:
                    N += 1
                    sumF += np.array(f[n]).astype('float64')

    u = sumF/float(N)

    for idx, ui in enumerate(u):
        if ui == 0.0:
            ui += 0.01
        fn[idx] = fn[idx]/float(ui)

    return fn


def preprocess(content):
    for line in content:
        extract(line)

    # FOR EACH NODE CALCULATE THE FEATURE TUPLE [ F0, F1, F2, F3, F4 ]

    for node in nodes:
        f[node] = [ ID(node), OD(node), IDW(node), ODW(node), LCC(node) ]

    # NORMALIZE

    for node in nodes:
        f[node] = normalize(f[node], node)


def train():
    pass

def test():
    pass


def main():
    content = read('./datasets/42.csv')

    preprocess(content[1:10])

    try:
        trained_model_file = open('trained_model.pickle', 'r')

        # USE THE TRAINED FILE

        # trained_model_file.close()

        # RUN TEST

        # test()
    except:
        train()





if __name__ == '__main__':
    main()
