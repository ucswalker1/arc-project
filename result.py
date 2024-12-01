import json


data = {}

for temp in [0, 0.2, 0.4, 0.6, 0.8]:
    with open(f'arc_results_{temp}.json', 'r') as f:
        data = json.load(f)


    total = 0
    correct = 0

    for entry in data:
        if entry["label"] == 1:
            correct += 1
        total += 1

    print (f"{correct / total} positive labels, {total} total \n")