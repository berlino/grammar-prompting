import os
import json
import random
from dataclasses import dataclass

from minEarley.parser import EarleyParser
from neural_lark.flags import FLAGS
from neural_lark.lark_utils import * 
from neural_lark.fol_utils import *
from third_party.structg.eval import check_equiv

@dataclass
class Example:
    source: str
    target: str

    grammar = None
    label = None

def load_examples(filename):
    examples = []
    assert len(filename.split(",")) == 2
    src_filename = filename.split(",")[0]
    trg_filename = filename.split(",")[1]
    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            examples.append(Example(source=line1.strip(), target=line2.strip(),))
    return examples

def load_sempar_data(config):
    """
    Note: 
    1. Uncomment the normalize_program_for_all function to normalize the programs, which would be useful for the baseline using linearized_tree (e.g., need to make sure programs can be predicted in a round-trip fashion)
    2. num_shot = -1 means using all the training data
    """

    def normalize_program_for_all(exs, parser):
        for ex in exs:
            try:
                normalized_target = normalize_program(ex.target, parser)
                ex.target = normalized_target
            except:
                logger.warning(f"failed to normalize program: {ex.target}")
                pass

    if config["dataset"] == "geoquery":
        split_name = config["split_name"]
        num_shot = config["num_shot"] 
        train_filename = f"data/geoquery/{split_name}/train.src,data/geoquery/{split_name}/train.tgt"
        dev_filename = f"data/geoquery/{split_name}/dev.src,data/geoquery/{split_name}/dev.tgt"
        test_filename = f"data/geoquery/{split_name}/test.src,data/geoquery/{split_name}/test.tgt"

        train_examples = load_examples(train_filename)
        dev_examples = load_examples(dev_filename)
        test_examples = load_examples(test_filename)

        # normalize_program_for_all(train_examples + dev_examples + test_examples, parser)

        if num_shot != -1:
            train_examples = train_examples[:num_shot]

    elif config["dataset"] == "smc":
        num_shot = config["num_shot"] 
        split_name = config["split_name"]

        # no need to escape 
        def unescape(ex):
            if "\"\\\"" in ex.target:
                new_target = ex.target.replace("\"\\\"", "\"")
                new_target = new_target.replace("\\\"", "")
                ex.target = new_target
        
        def unescape_all(examples):
            for ex in examples:
                unescape(ex)
            
        if split_name == "indomain":
            data_dir = f"data/smcalflow_cs/calflow.orgchart.event_create_v2/source_domain_with_target_num0"
            train_filename = f"{data_dir}/train.canonical.src,{data_dir}/train.canonical.tgt"
            dev_filename = f"{data_dir}/valid.canonical.indist.src,{data_dir}/valid.canonical.indist.tgt"
            test_filename = f"{data_dir}/test.canonical.indist.src,{data_dir}/test.canonical.indist.tgt"

            train_examples = load_examples(train_filename)
            dev_examples = load_examples(dev_filename)
            test_examples = load_examples(test_filename)
            unescape_all(train_examples + dev_examples + test_examples)

            # normalize_program_for_all(train_examples + dev_examples + test_examples, parser)

            if num_shot != -1:
                train_examples = train_examples[:num_shot]

        else:
            assert split_name == "comp"
            data_dir = f"data/smcalflow_cs/calflow.orgchart.event_create_v2/source_domain_with_target_num{num_shot}"
            train_filename = f"{data_dir}/train.canonical.src,{data_dir}/train.canonical.tgt"
            dev_filename = f"{data_dir}/valid.canonical.outdist.src,{data_dir}/valid.canonical.outdist.tgt"
            test_filename = f"{data_dir}/test.canonical.outdist.src,{data_dir}/test.canonical.outdist.tgt"

            train_examples = load_examples(train_filename)
            dev_examples = load_examples(dev_filename)
            test_examples = load_examples(test_filename)
            unescape_all(train_examples + dev_examples + test_examples)
            # normalize_program_for_all(train_examples + dev_examples + test_examples, parser)

            if num_shot != -1:
                logger.info(f"Only use {num_shot} comp examples")
                train_examples = train_examples[-num_shot:]
    
    elif config["dataset"] == "overnight":
        domain = config["domain"]
        num_shot = config["num_shot"]
        data_dir = f"data/overnight/{domain}"
        train_filename = f"{data_dir}/train.src,{data_dir}/train.tgt"
        dev_filename = f"{data_dir}/dev.src,{data_dir}/dev.tgt"
        test_filename = f"{data_dir}/test.src,{data_dir}/test.tgt"

        train_examples = load_examples(train_filename)
        dev_examples = load_examples(dev_filename)
        test_examples = load_examples(test_filename)

        if num_shot != -1:
            train_examples = train_examples[-num_shot:]

    elif config["dataset"] == "mtop":
        split_name = config["split_name"]
        num_shot = config["num_shot"]
        data_dir = f"data/mtop/{split_name}-numshot{num_shot}"
        train_filename = f"{data_dir}/train.src,{data_dir}/train.tgt"
        dev_filename = f"{data_dir}/dev.src,{data_dir}/dev.tgt"
        test_filename = f"{data_dir}/test.src,{data_dir}/test.tgt"

        train_examples = load_examples(train_filename)
        dev_examples = load_examples(dev_filename)
        test_examples = load_examples(test_filename)

        if "indomain" in split_name and num_shot != -1:
            train_examples = train_examples[:num_shot]
        else:
            assert num_shot == -1
    
    elif config["dataset"] == "regex":
        num_shot = config["num_shot"]
        data_dir = f"data/regex/fewshot_num{num_shot}"
        train_filename = f"{data_dir}/train.src,{data_dir}/train.tgt"
        dev_filename = f"{data_dir}/valid.src,{data_dir}/valid.tgt"
        test_filename = f"{data_dir}/testi.src,{data_dir}/testi.tgt"

        train_examples = load_examples(train_filename)
        dev_examples = load_examples(dev_filename)
        test_examples = load_examples(test_filename)

        # normalize_program_for_all(train_examples + dev_examples + test_examples, parser)
    elif config["dataset"] == "folio":
        def load_fol_examples(filename):
            examples = []
            with open(filename) as f:
                for line in f:
                    example = json.loads(line)

                    nl_l = example["premises"] + [example["conclusion"]]
                    nl = "\n".join(nl_l)

                    fol_l = example["premises-FOL"] + [example["conclusion-FOL"]]
                    fol = "\n".join(fol_l)

                    label = example["label"]

                    example = Example(nl, fol)
                    example.label = label
                    examples.append(example)
            return examples
        
        # train_filename = f"data/folio/folio-train.jsonl"
        dev_filename = f"data/folio/folio-validation.jsonl"
        orig_dev_examples = load_fol_examples(dev_filename)
        random.shuffle(orig_dev_examples)

        assert config["num_shot"] != -1
        train_examples = orig_dev_examples[:config["num_shot"]]
        dev_examples = []
        test_examples = orig_dev_examples[config["num_shot"]:]

    else:
        raise ValueError(f"dataset {config['dataset']} not supported")

    ## dryrun mode
    if getattr(FLAGS, "dryrun", False):
        os.environ["WANDB_MODE"] = "dryrun"
        batch_size = 20
        dev_examples = dev_examples[:batch_size // 2]
        test_examples = test_examples[:batch_size // 2]
    
    if getattr(FLAGS, "quickrun", False):
        os.environ["WANDB_MODE"] = "dryrun"
        test_examples = test_examples[:100]
    
    logger.info(f"num train examples: {len(train_examples)}, num dev examples: {len(dev_examples)}, num test examples: {len(test_examples)}")
    
    return train_examples, dev_examples, test_examples

def load_sem_parser(config):
    if config["dataset"] == "geoquery":
        grammar_file = "grammars/geo.lark"
        global_parser = EarleyParser.open(grammar_file, start='query', keep_all_tokens=True)
    elif config["dataset"] == "smc":
        # grammar_file, start_symbol = "grammars/lispress_full_1.lark", "list"
        grammar_file, start_symbol = "grammars/lispress_full_3.lark", "call"
        global_parser = EarleyParser.open(grammar_file, start=start_symbol, keep_all_tokens=True)
    elif config["dataset"] == "mtop":
        domain = config["split_name"].split("-")[1]
        grammar_file, start_symbol = f"grammars/mtop/{domain}.lark", "query"
        global_parser = EarleyParser.open(grammar_file, start=start_symbol, keep_all_tokens=True)
    elif config["dataset"] == "regex":
        # grammar_file = "grammars/regex_simple.lark"
        grammar_file = "grammars/regex_medium.lark"
        # grammar_file = "grammars/regex_hard.lark"
        global_parser = EarleyParser.open(grammar_file, start='regex', keep_all_tokens=True)
    elif config["dataset"] == "overnight":
        domain = config["domain"]
        grammar_file, start_symbol = f"grammars/overnight/{domain}.lark", "list_value"
        global_parser = EarleyParser.open(grammar_file, start=start_symbol, keep_all_tokens=True)
    elif config["dataset"] == "folio":
        grammar_file, start_symbol = f"grammars/fol.lark", "formula"
        global_parser = EarleyParser.open(grammar_file, start=start_symbol, keep_all_tokens=True)
    else:
        raise ValueError(f"dataset {config['dataset']} not supported")
    global_rules, _ = collect_rules_from_larkfile(grammar_file)
    return global_parser, global_rules

def counter2pred(counter):
    if len(counter) == 0:
        return None
    else:
        return counter.most_common(1)[0][0]

def evaluate_programs(predictions, examples):
    if len(examples) == 0:
        return 0.0

    counter = 0
    for prediction_counter, example in zip(predictions, examples):
        prediction = counter2pred(prediction_counter) 
        if prediction == example.target:
            counter += 1
    return counter / len(predictions)


def evaluate_grammars(grammars, examples, global_parser):
    if grammars is None or len(grammars) == 0:
        return 0.0
    
    counter = 0
    for grammar_counter, example in zip(grammars, examples):
        if grammar_counter is None or len(grammar_counter) == 0:
            continue

        lark_grammar = counter2pred(grammar_counter)
        try:
            parse_tree = global_parser.parse(example.target)
        except Exception as e:
            logger.warning(f"failed to parse target program:\n{example.target}")
            continue
        _, min_rules = extract_min_grammar_from_trees(parse_tree, return_rules=True)
        if check_grammar_correctness(min_rules, lark_grammar):
            counter += 1
    return counter / len(grammars)


def evaluate_dfa(predictions, examples):
    def unnaturalize(text):
        text = text.replace("notcontain", "notcc")
        text = text.replace("<letter>", "<let>")
        text = text.replace("<lowercase>", "<low>")
        text = text.replace("<uppercase>", "<cap>")
        text = text.replace("<number>", "<num>")
        text = text.replace("<special>", "<spec>")
        text = text.replace("constant(", "const(")
        return text

    if len(examples) == 0:
        return 0.0

    counter = 0
    for prediction_counter, example in zip(predictions, examples):
        prediction = counter2pred(prediction_counter) 
        if prediction:
            prediction = unnaturalize(prediction)
            target = unnaturalize(example.target)
            if check_equiv(prediction, target):
                counter += 1
    return counter / len(predictions)

def evaluate_fol(predictions, examples, parser):
    if len(examples) == 0:
        return 0.0
    
    counter = 0
    for prediction_counter, example in zip(predictions, examples):
        prediction = counter2pred(prediction_counter) 

        try:
            logger.disabled = True
            prediction = cleanup_fol(prediction)
            pred_premises, pred_hyp = parse_fol(prediction, parser)
            pred_res = execute_fol(pred_premises, pred_hyp)

            if isinstance(pred_res, dict):
                all_vals = list(set(v.responseStr() for v in pred_res.values()))
                if len(all_vals) == 1:
                    pred_val = all_vals[0]
                elif "Yes." in all_vals:
                    pred_val = "Yes."
                else:
                    pred_val = "I don't know."
            else:
                pred_val = pred_res.responseStr()
            logger.disabled = False
        except Exception as e:
            logger.warning(f"failed to execute predicted program:\n{prediction}")
            logger.warning(str(e))
            pred_val = "I don't know."

        tgt_premises, tgt_hyp = parse_fol(example.target, parser)
        tgt_res = execute_fol(tgt_premises, tgt_hyp)
        tgt_val = tgt_res.responseStr()

        logger.info(f"NL sentences:\n{example.source}")
        logger.info(f"predicted program:\n{prediction}")
        logger.info(f"predicted value: {pred_val}")
        logger.info(f"target program:\n{example.target}")
        logger.info(f"target value: {tgt_val}")

        if pred_val == tgt_val:
            counter += 1
    return counter / len(predictions)