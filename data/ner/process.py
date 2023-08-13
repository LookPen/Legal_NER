import json

path = "./xxcq_mid.json"

data = []
ner_types = []
for line in open(path).readlines():
    line = json.loads(line)
    new_line = {
        "text": line["context"],
        "entities": []
    }
    for ent in line["entities"]:
        if len(ent["span"]):
            for item in ent["span"]:
                s,e = item.split(";")
                s = int(s)
                e = int(e)
                new_line["entities"].append({
                    "entity": line["context"][s:e],
                    "start_idx":s,
                    "end_idx":e-1,
                    "type": ent["label"]
                })
            ner_types.append(ent["label"])
    data.append(new_line)


import random
random.shuffle(data)
train = data[:4500]
test = data[4500:]
json.dump(
    {"data": train},
    open("./train.json", "w"),
    ensure_ascii=False,
    indent=4
)

json.dump(
    {"data":test},
    open("./test.json", "w"),
    ensure_ascii=False,
    indent=4
)
ner_types = list(set(ner_types))
json.dump(ner_types, open("./ent_types.json", "w"), ensure_ascii=False, indent=4)
