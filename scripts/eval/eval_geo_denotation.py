import sys
import json

from third_party.geo_eval.executor import ProgramExecutorGeo

def dict2str(d):
    return list(d.keys())[0]

def execute(programs):
    executor = ProgramExecutorGeo()
    denotations = []
    for program in programs:
        denotation = executor.execute(program)
        if denotation.startswith("error_parse:"):
            denotation = None
        denotations.append(denotation)
    return denotations

pred_json = sys.argv[1]
split = sys.argv[2]

test_predictions = []
with open(pred_json, "r") as f:
    pred_dict = json.load(f)
    test_prediction_dicts = pred_dict["test_predictions"]
    for p in test_prediction_dicts:
        p = dict2str(p)
        test_predictions.append(p)

ref_file = f"data/geoquery/{split}_split/test.tgt"

references = []
with open(ref_file, "r") as f:
    for line in f:
        line = line.strip()
        ref_lf = line
        references.append(ref_lf)

pred_denotations = execute(test_predictions)
ref_denotations = execute(references)

if len(pred_denotations) < len(ref_denotations):
    print("WARNING: evaluting on a subset ")
# assert len(pred_denotations) == len(ref_denotations)

counter = 0
for pred_d, ref_d, pred, ref in zip(pred_denotations, ref_denotations, test_predictions, references):
    if pred_d == ref_d:
        counter += 1

        if pred != ref:
            print("pred: ", pred)
            print("ref: ", ref)
            print("pred_d: ", pred_d)
            print("ref_d: ", ref_d)
            print()
print(counter / len(pred_denotations))
