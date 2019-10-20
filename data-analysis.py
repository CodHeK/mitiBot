import re, random

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

with open('datasets/42.csv', 'r') as file:
    whole = file.readlines()

train, test = mod(whole[1:])

flow_set = {}
node_set = {}

bot_flow = 0
non_bot_flow = 0

# print("Lines in Dataset: ", str(len(whole[1:])))

for x in train:
    sip = x.split(',')[3]
    dip = x.split(',')[6]

    if sip not in node_set:
        node_set[sip] = 1
    if dip not in node_set:
        node_set[sip] = 1

    flow = x.split(',')[14]

    res = re.findall("Botnet", str(flow))

    if len(res) > 0:
        bot_flow += 1
    else:
        non_bot_flow += 1

    if flow not in flow_set:
        flow_set[flow] = 1
    else:
        flow_set[flow] += 1


# print("Number of nodes: ", str(len(node_set)))

print("Total type of Flows: ", str(len(flow_set)))

botnet_type_flows = 0

for flow in flow_set:
    if len(re.findall("Botnet", str(flow))) > 0:
        botnet_type_flows += 1

print("Total type of Botnet Flows: ", str(botnet_type_flows))

print("------------------------------------")

print("Percentage type of botnet flows in dataset = ", str((botnet_type_flows*100)/float(len(flow_set))) + " %")

print("------------------------------------")

print("Non-Botnet Flows: ", str(non_bot_flow))

print("Botnet Flows: ", str(bot_flow))

print("------------------------------------")

print("Percentage botnet entries in datasets = ", str((bot_flow*100)/float(non_bot_flow + bot_flow)) + " %")

print("------------------------------------")
