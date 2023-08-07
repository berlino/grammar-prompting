import json
import sys

# json_file = "length_split/length_1.json"
# train_file = "length_split/train"
# test_file = "length_split/test"

# json_file = "template_split/template_1.json"
# train_file = "template_split/train"
# test_file = "template_split/test"

json_file = "tmcd_split/tmcd_1.json"
train_file = "tmcd_split/train"
test_file = "tmcd_split/test"

with open(json_file) as f:
    data = json.load(f)

with open(train_file, "w") as f:
    for idx in data["train"]:
        f.write(f"{idx}\n")

with open(test_file, "w") as f:
    for idx in data["test"]:
        f.write(f"{idx}\n")
