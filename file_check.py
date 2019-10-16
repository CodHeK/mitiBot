with open('42.csv', 'r') as file:
    whole = file.readlines()

flow_set = {}
node_set = {}

print("Lines in Dataset: ", str(len(whole[1:])))

for x in whole[1:]:
    sip = x.split(',')[3]
    dip = x.split(',')[6]

    if sip not in node_set:
        node_set[sip] = 1
    if dip not in node_set:
        node_set[sip] = 1

    flow = x.split(',')[14]

    if flow not in flow_set:
        flow_set[flow] = 1


print("Number of nodes: ", str(len(node_set)))

print("Number of flows: ", str(len(flow_set)))

for f in flow_set:
    print(f)
