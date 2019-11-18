import re

def read(files):
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


def mod(content):
    non_bot_tuples = []
    bot_tuples = []

    non_bot_flow_tuples = 100000

    for line in content:
        flow = line.split(',')[14]

        # NON BOTNET FLOWS
        if len(re.findall("Botnet", str(flow))) == 0:
            if non_bot_flow_tuples > 0:
                non_bot_flow_tuples -= 1
                non_bot_tuples.append(line)
        else:
            bot_tuples.append(line)

    per = (len(bot_tuples)*100)/float(len(non_bot_tuples) + len(bot_tuples))
    print("mod = " + str(per))

    return (non_bot_tuples, bot_tuples)


def nomod(content):
    non_bot_tuples = []
    bot_tuples = []

    for line in content:
        flow = line.split(',')[14]

        # NON BOTNET FLOWS
        if len(re.findall("Botnet", str(flow))) == 0:
            non_bot_tuples.append(line)
        else:
            bot_tuples.append(line)

    per = (len(bot_tuples)*100)/float(len(non_bot_tuples) + len(bot_tuples))
    print("no mod = " + str(per))

    return (non_bot_tuples, bot_tuples)


content = read(['42.csv', '43.csv', '46.csv', '47.csv', '48.csv', '50.csv', '51.csv', '52.csv', '53.csv'])

mod(content)
nomod(content)
