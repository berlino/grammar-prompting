import os
from neural_lark.overnight_utils import normalize_lf

domain = "blocks"

all_train_src, all_train_tgt = [], []
with open(f"data/overnight/raw/{domain}_train.tsv") as f:
    for line in f:
        line = line.strip()
        nl, program, _ = line.split("\t")
        program = normalize_lf(program)
        all_train_src.append(nl)
        all_train_tgt.append(program)

split_point = int(len(all_train_src) * 0.9)
train_src = all_train_src[:split_point]
train_tgt = all_train_tgt[:split_point]
dev_src = all_train_src[split_point:]
dev_tgt = all_train_tgt[split_point:]

test_src, test_tgt = [], []
with open(f"data/overnight/raw/{domain}_test.tsv") as f:
    for line in f:
        line = line.strip()
        nl, program, _ = line.split("\t")
        program = normalize_lf(program)
        test_src.append(nl)
        test_tgt.append(program)

if not os.path.exists(f"data/overnight/{domain}"):
    os.mkdir(f"data/overnight/{domain}")

with open(f"data/overnight/{domain}/train.src", "w") as f:
    for src in train_src:
        f.write(src + "\n")
with open(f"data/overnight/{domain}/train.tgt", "w") as f:
    for tgt in train_tgt:
        f.write(tgt + "\n")
with open(f"data/overnight/{domain}/dev.src", "w") as f:
    for src in dev_src:
        f.write(src + "\n")
with open(f"data/overnight/{domain}/dev.tgt", "w") as f:
    for tgt in dev_tgt:
        f.write(tgt + "\n")
with open(f"data/overnight/{domain}/test.src", "w") as f:
    for src in test_src:
        f.write(src + "\n")
with open(f"data/overnight/{domain}/test.tgt", "w") as f:
    for tgt in test_tgt:
        f.write(tgt + "\n")