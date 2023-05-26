import sys
import json
from neural_lark.overnight_utils import denormalize_lf, execute

def dict2str(d):
    return list(d.keys())[0]

pred_json = sys.argv[1]

test_predictions = []
with open(pred_json, "r") as f:
    pred_dict = json.load(f)
    test_prediction_dicts = pred_dict["test_predictions"]
    for p in test_prediction_dicts:
        p = dict2str(p)
        try:
            p = denormalize_lf(p)
        except:
            pass
        test_predictions.append(p)

ref_file = "data/overnight/blocks/test.tgt"
references = []
with open(ref_file, "r") as f:
    for line in f:
        line = line.strip()
        ref_lf = denormalize_lf(line)
        references.append(ref_lf)

pred_denotations = execute(test_predictions, "blocks")
ref_denotations = execute(references, "blocks")

counter = 0
for pred, ref in zip(pred_denotations, ref_denotations):
    if pred == ref:
        counter += 1
print(counter / len(pred_denotations))
