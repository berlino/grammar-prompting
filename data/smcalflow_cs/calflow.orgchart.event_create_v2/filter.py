# filter in-dist vs out-dist

num_shot = 0
data_dir = f"source_domain_with_target_num{num_shot}"

dev_src_file = f"{data_dir}/valid.canonical.src"
dev_tgt_file = f"{data_dir}/valid.canonical.tgt"
dev_indist_src_file = f"{data_dir}/valid.canonical.indist.src"
dev_indist_tgt_file = f"{data_dir}/valid.canonical.indist.tgt"
dev_outdist_src_file = f"{data_dir}/valid.canonical.outdist.src"
dev_outdist_tgt_file = f"{data_dir}/valid.canonical.outdist.tgt"

with open(dev_src_file, "r") as src_f, open(dev_tgt_file, "r") as tgt_f, \
    open(dev_indist_src_file, "w") as indist_src, open(dev_indist_tgt_file, "w") as indist_tgt, \
    open(dev_outdist_src_file, "w") as outdist_src, open(dev_outdist_tgt_file, "w") as outdist_tgt:
    for src, tgt in zip(src_f, tgt_f):
        if ("FindManager" in tgt or "FindReports" in tgt or "FindTeamOf" in tgt) and ("Event" in tgt):
            outdist_src.write(src)
            outdist_tgt.write(tgt)
        else:
            indist_src.write(src)
            indist_tgt.write(tgt)

test_src_file = f"{data_dir}/test.canonical.src"
test_tgt_file = f"{data_dir}/test.canonical.tgt"
test_indist_src_file = f"{data_dir}/test.canonical.indist.src"
test_indist_tgt_file = f"{data_dir}/test.canonical.indist.tgt"
test_outdist_src_file = f"{data_dir}/test.canonical.outdist.src"
test_outdist_tgt_file = f"{data_dir}/test.canonical.outdist.tgt"

with open(test_src_file, "r") as src_f, open(test_tgt_file, "r") as tgt_f, \
    open(test_indist_src_file, "w") as indist_src, open(test_indist_tgt_file, "w") as indist_tgt, \
    open(test_outdist_src_file, "w") as outdist_src, open(test_outdist_tgt_file, "w") as outdist_tgt:
    for src, tgt in zip(src_f, tgt_f):
        if ("FindManager" in tgt or "FindReports" in tgt or "FindTeamOf" in tgt) and ("Event" in tgt):
            outdist_src.write(src)
            outdist_tgt.write(tgt)
        else:
            indist_src.write(src)
            indist_tgt.write(tgt)
