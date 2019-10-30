import numpy as np
import pprint, re, random, pickle, json, argparse
from datetime import datetime

class Build:
    def __init__(self, files):
        self.data = []
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
        self.content = self.read(files)
        self.non_bot_tuples, self.bot_tuples = self.mod(self.content)


    '''
        File read helper function
    '''
    def read(self, files):
        combined_content = []
        for file in files:
            with open('./datasets/' + str(file), 'r') as of:
                content = of.readlines()
                for line in content[1:]:
                    combined_content.append(line)
            of.close()
            print("Read file - " + str(file))

        print("Read Dataset...")

        return combined_content


    '''
        Return tuples containing non-bot flows and bot-flows separately
    '''
    def mod(self, content):
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

        return (non_bot_tuples, bot_tuples)


    def build_train_set(self, nb, b):
        train_set = nb + b
        random.shuffle(train_set)

        print("Built training dataset...")

        return train_set


    def build_test_set(self, nb, b, p):
        print("Total tuples to choose test set from: " + str(len(nb + b)) + " = " + str(len(nb)) + " + " + str(len(b)))

        tnb = int((p*len(nb))/100)

        random.shuffle(nb)
        random.shuffle(b)

        test_set = nb[:tnb] + b

        print("Total non-bot tuples: " + str(tnb) + " -> " + str((tnb*100)/float(tnb+len(b))) + " %")

        print("Total bot tuples: " + str(len(b)) + " -> " + str((len(b)*100)/float(tnb+len(b))) + " %")

        random.shuffle(test_set)

        print("Total size of test set: " + str(tnb + len(b)))

        print("Built testing dataset...")

        return test_set


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
                    self.graph[dip][sip] = (srcpktsY + dstpktsX, (srcpkts, dstpkts))

            return (sip, dip)


    '''
        F-Norm Implementation using D = 1
    '''
    def normalize(self, fn, node):
        N = 0
        sumF = np.array([0, 0, 0, 0]).astype('float64')
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

        node_dict = {}
        for n in self.node_map:
            if self.node_map[n] not in node_dict:
                node_dict[self.node_map[n]] = 1
            else:
                node_dict[self.node_map[n]] += 1

        print(node_dict)

        print("Graph built!")

        # FOR EACH NODE CALCULATE THE FEATURE TUPLE [ F0, F1, F2, F3, F4 ]

        # bots = 0
        # for it in self.node_map:
        #     if self.node_map[it] == 1:
        #         bots += 1

        print(len(self.nodes))
        # print(bots)

        for node in self.nodes:
            self.f[node] = [ [ self.ID(node), self.OD(node), self.IDW(node), self.ODW(node) ], self.node_map[node] ]

        print("Feature Extraction done!")

        with open('./saved/f.json', 'w') as feat:
            json.dump(self.f, feat, indent=4)

        # NORMALIZE

        for node in self.nodes:
            # self.f[node][0] = self.normalize(self.f[node][0], node)
            self.fvecs.append(self.f[node][0])
            # if self.node_map[node] == 1:
            #     print(self.f[node][0])

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
