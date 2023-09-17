import json

# 0917 train2.json 在原有json的基础上添加ent_types内容
data = json.load(open("./train.json", encoding="utf-8"))["data"]

new_data = []
for example in data:
    ent_types = []

    for ent in example["entities"]:
        if ent["type"] not in ent_types:
            ent_types.append(ent["type"])

    example["ent_types"] = ent_types
    new_data.append(example)

json.dump({"data": new_data}, open("./train2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)

# 0917 test2.json 中ent_types的内容来自分类器
with open(r'predict_results_None.txt') as f:
    lines = f.readlines()
    all_pred = []
    for i in range(1, len(lines)):
        pred_types = json.loads(lines[i])['pred']
        all_pred.append(pred_types)

    data = json.load(open("./test.json", encoding="utf-8"))["data"]
    new_data = []

    for i in range(len(data)):
        example = data[i]
        example["ent_types"] = all_pred[i]
        new_data.append(example)

    json.dump({"data": new_data}, open("./test2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)
