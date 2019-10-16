# GLOBAL VALUES

graph = {}
nodes = {}


def read(filename):
    with open(filename, 'r') as file:
        content = file.readlines()

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

def preprocess(content):
    for line in content:
        (sip, dip) = extract(line)



def train():
    pass


def test():
    pass




def main():
    content = read('./datasets/42.csv')

    preprocess(content[1:10])


if __name__ == '__main__':
    main()
