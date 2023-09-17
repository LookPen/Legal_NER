import json

data = json.load(open("./test.json", encoding="utf-8"))["data"]

new_data = []
for example in data:
    ent_types = []

    for ent in example["entities"]:
        if ent["type"] not in ent_types:
            ent_types.append(ent["type"])

    example["ent_types"] = ent_types
    new_data.append(example)

# 0917 train2.json 在原有json的基础上添加ent_types内容，test2.json 中ent_types的内容来自分类器
json.dump({"data": new_data}, open("./train2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)