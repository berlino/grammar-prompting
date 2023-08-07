import os
import json
import time
import random
import tqdm

import shutil
import wandb
import functools
import collections
from typing import List
from dataclasses import dataclass

import numpy as np
from minEarley.parser import EarleyParser

from neural_lark.llm_interface import setup_llm 
from neural_lark.train_utils import logger, setup_logger_file
from neural_lark.flags import FLAGS, parse_args
from neural_lark.lark_utils import lark2bnf, bnf2lark, decorate_grammar, extract_rule_stat

import fcntl
from fuseprop.chemutils import get_mol, get_smiles
from rdkit import DataStructs, Chem
from rdkit.Chem import AllChem

@dataclass
class Example:
    molecule: str
    grammar = None

def load_dataset(config):
    def load_examples(filename):
        examples = []
        with open(filename) as f:
            for line in f:
                molecule = line.strip()
                examples.append(Example(molecule))
        return examples
    
    dataset_filename = f"data/deg/{config['dataset']}.txt"
    return load_examples(dataset_filename)

def load_parser(config, use_generic_grammar=False):
    if use_generic_grammar:
        grammar_filename = f"grammars/deg/generic.lark"
    else:
        grammar_filename = f"grammars/deg/{config['dataset']}.lark"

    global_parser = EarleyParser.open(grammar_filename, start="smiles", keep_all_tokens=True)
    global_grammar_str = ""
    for line in open(grammar_filename):
        line = line.rstrip()
        line = line.replace(": ", " ::= ")
        if line:
            global_grammar_str += line + "\n"
    return global_parser, global_grammar_str

def extract_min_grammar(molecule, parser):
    parse_tree = parser.parse(molecule)
    rule_stat = collections.OrderedDict()
    extract_rule_stat(parse_tree, rule_stat)
    
    lhs2rhs = collections.OrderedDict()
    for rule in rule_stat:
        lhs, rhs = rule.origin, rule.expansion
        if lhs not in lhs2rhs:
            lhs2rhs[lhs] = []
        lhs2rhs[lhs].append(rhs)
    
    grammar = ""
    for lhs in lhs2rhs:
        grammar += f"{lhs} :"
        rhs_str = ""
        for rhs in lhs2rhs[lhs]:
            one_rhs_str = " ".join(rhs)
            rhs_str += f" {one_rhs_str} |"
        rhs_str = rhs_str[:-2]
        grammar += rhs_str
        grammar += "\n"
    
    return grammar.strip()

DELIMITER = "\nMolecule based on the BNF grammar rules: "
prompt_template = {
    "std": {
        "instruction": "You are an expert in chemistry. You are given a list of {name} molecules in SMILES format. You are asked to write another {name} molecule in SMILES format.\n",
        "exemplar": lambda ex: f"Molecule: {ex.molecule}\n",
        "prediction": lambda ex: "Molecule: ",
    },
    "wrule": {
        "instruction": "You are an expert in chemistry. You are given a list of molecules in SMILES format. You are asked to write another molecule of the same class in SMILES format. To accomplish this, you will first need to synthesize a set of rules that can be used to generate these molecules. Once you have derived the necessary rules, apply them to create a new molecule that belongs to the same class.\n",
        "exemplar": lambda ex: f"BNF grammar rules:\n{ex.grammar}{DELIMITER}{ex.molecule}\n\n",
        "prediction": lambda ex: "BNF grammar rules:\n",
        "prediction_given_rule": lambda g: f"BNF grammar rules:\n{g}{DELIMITER}",
    },
    "wrule_and_mname": {
        "instruction": "You are an expert in chemistry. You are given a list of {name} molecules in SMILES format. You are asked to write another {name} molecule in SMILES format. To accomplish this, you will first need to synthesize a set of rules that can be used to generate these molecules. Once you have derived the necessary rules, apply them to create a new {name} molecule.\n", 
        "exemplar": lambda ex: f"BNF grammar rules:\n{ex.grammar}{DELIMITER}{ex.molecule}\n\n",
        "prediction": lambda ex: "BNF grammar rules:\n",
        "prediction_given_rule": lambda g: f"BNF grammar rules:\n{g}{DELIMITER}",
    }
}

def batch_prompt_predict(llm, exemplars, prompt_template, num_samples):
    template_ex = prompt_template["exemplar"]
    template_p = prompt_template["prediction"]

    samples = set()
    exemplar_mols = [ex.molecule for ex in exemplars]
    while len(samples) < num_samples:
        random.shuffle(exemplars)
        fewshot_prompt = prompt_template["instruction"]
        for ex in exemplars:
            fewshot_prompt += template_ex(ex)
        fewshot_prompt += template_p(None)

        response = llm.sample_completions(fewshot_prompt, temperature=FLAGS.temperature, stop_token="\n", disable_cache=True)[0]
        new_sample = response.response_text
        if new_sample and new_sample not in samples and new_sample not in exemplar_mols:
            logger.info(f"Generated Molecule: {new_sample} with index {len(samples)}")
            samples.add(new_sample)
        else:
            logger.info(f"Generated Molecule: {new_sample} (duplicate)")
    return list(samples)

def having_too_many_repetition(rule_str):
    rule_tokens = rule_str.split()
    counter = collections.Counter(rule_tokens)
    if counter.most_common(1)[0][1] > 32:
        most_freq_token = counter.most_common(1)[0][0]
        logger.warning(f"Rule {rule_str} contains token {most_freq_token} more than 32 times. Skip.")
        return False
    else:
        return True

def predict_program_with_earley_correction(prompt, parser):
    """
    Args:
        prompt: prompt for Codex
    """
    MAX_NUM_CORRECTION = 6
    num_correction_left = MAX_NUM_CORRECTION

    def validate_program(prediction):
        try:
            parser.parse(prediction)
            return True
        except Exception as runtime_e:
            logger.info(f"Error in prediction: {prediction}")
            logger.info(f"Error: {str(runtime_e)}")
            return False
        
    def obtain_correction_pairs(prediction):
        """
        Returns a list of candidates in the form of (prefix, suffix).
        """
        try:
            parser.parse(prediction)
            return []
        except Exception as runtime_e:
            return parser.handle_error(runtime_e)

    partial_program_prediction = ""
    ret_prediction, initial_prediction = None, None
    while num_correction_left > 0:
        _prompt = prompt + partial_program_prediction
        residual_program_prediction = llm_interface.run_completion(_prompt)[0]

        # if the prediction is empty, return the initial prediction
        if initial_prediction is None:
            initial_prediction = residual_program_prediction
        program_prediction = partial_program_prediction + residual_program_prediction

        if validate_program(program_prediction):
            ret_prediction = program_prediction
            break

        # find the max score from a list of score
        pairs = obtain_correction_pairs(program_prediction)
        assert len(pairs) > 0, "no correction pairs found"
        logger.info(f"number of candidates: {len(pairs)}")
        best_idx = random.randint(0, len(pairs) - 1)
        fixed_prediction = pairs[best_idx][0] + pairs[best_idx][1]
        logger.warning(f"fixed prediction: {fixed_prediction}")

        partial_program_prediction = fixed_prediction
        num_correction_left -= 1

    return ret_prediction

def batch_prompt_predict_with_rule(llm, exemplars, prompt_template, num_samples, constrain_prog):
    def postprocess_sample(sample):
        sample = sample.split("\n")[0]
        sample = sample.replace("<|im_end|>", "")
        return sample

    template_ex = prompt_template["exemplar"]
    template_p = prompt_template["prediction"]
    template_p_given_rule = prompt_template["prediction_given_rule"]

    samples = set()
    exemplar_mols = [ex.molecule for ex in exemplars]
    while len(samples) < num_samples:
        random.shuffle(exemplars)
        fewshot_prompt = prompt_template["instruction"]
        for ex in exemplars:
            min_grammar_str = extract_min_grammar(ex.molecule, global_parser)
            ex.grammar = lark2bnf(min_grammar_str)
            fewshot_prompt += template_ex(ex)
        
        # 1. rule prediction
        fewshot_prompt_for_rule = fewshot_prompt + template_p(None)
        response = llm.sample_completions(fewshot_prompt_for_rule, temperature=FLAGS.rule_temperature, stop_token=DELIMITER, disable_cache=True)[0]
        rule_prediction = response.response_text
        logger.info(f"Generated Rule:\n{rule_prediction}")

        if not having_too_many_repetition(rule_prediction):
            continue

        # 2. program prediction
        fewshot_prompt_for_program = fewshot_prompt + template_p_given_rule(rule_prediction)
        response = llm.sample_completions(fewshot_prompt_for_program, temperature=FLAGS.temperature, stop_token="\n\n", disable_cache=True)[0]
        new_sample = response.response_text
        new_sample = postprocess_sample(new_sample)

        if constrain_prog:
            try:
                logger.info(f"Generated Molecule for verification: {new_sample}")

                local_parser = EarleyParser(decorate_grammar(bnf2lark(rule_prediction)), start=global_parser.option.start)
                _ = local_parser.parse(new_sample)
            except Exception as e:
                logger.warning(f"Error: {str(e)}")
                continue
        
        if new_sample and new_sample not in samples and new_sample not in exemplar_mols:
            logger.info(f"Generated Molecule: {new_sample} with index {len(samples)}")
            samples.add(new_sample)
        else:
            logger.info(f"Generated Molecule: {new_sample} (duplicate)")
    return list(samples)

class InternalDiversity():
    def distance(self, mol1, mol2, dtype="Tanimoto"):
        assert dtype in ["Tanimoto"]
        if dtype == "Tanimoto":
            sim = DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(mol1), Chem.RDKFingerprint(mol2))
            return 1 - sim
        else:
            raise NotImplementedError

    def get_diversity(self, mol_list, dtype="Tanimoto"):
        similarity = 0
        mol_list = [AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048) for x in mol_list] 
        for i in range(len(mol_list)):
            sims = DataStructs.BulkTanimotoSimilarity(mol_list[i], mol_list[:i])
            similarity += sum(sims)
        n = len(mol_list)
        n_pairs = n * (n - 1) / 2
        diversity = 1 - similarity / n_pairs
        return diversity

def retro_sender(generated_samples, sender_file="log/deg/generated_samples.txt", receiver_file="log/deg/output_syn.txt"):
    def lock(f):
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except IOError:
            return False
        return True

    # File communication to obtain retro-synthesis rate
    with open(receiver_file, 'w') as fw:
        fw.write('')
    while(True):
        with open(sender_file, 'r') as fr:
            editable = lock(fr)
            if editable:
                with open(sender_file, 'w') as fw:
                    for sample in generated_samples:
                        fw.write('{}\n'.format(Chem.MolToSmiles(sample)))
                break
            fcntl.flock(fr, fcntl.LOCK_UN)
    num_samples = len(generated_samples)
    print("Waiting for retro_star evaluation...")
    while(True):
        with open(receiver_file, 'r') as fr:
            editable = lock(fr)
            if editable:
                syn_status = []
                lines = fr.readlines()
                if len(lines) == num_samples:
                    for idx, line in enumerate(lines):
                        splitted_line = line.strip().split()
                        syn_status.append((idx, splitted_line[2]))
                    break
            fcntl.flock(fr, fcntl.LOCK_UN)
        time.sleep(1)
    assert len(generated_samples) == len(syn_status)
    return np.mean([int(eval(s[1])) for s in syn_status])

def eval_membership(mol_sample, monomer_class):
    if monomer_class == "acrylates":
        patterns = ["C=CC(=O)O*"]
        pattern_mols = [Chem.MolFromSmarts(p) for p in patterns]
        for pattern in pattern_mols:
            if mol_sample.HasSubstructMatch(pattern):
                return True
        return False
    elif monomer_class == "chain_extenders":
        patterns = ['CO', 'OC', 'N']
        pattern_mols = [Chem.MolFromSmarts(p) for p in patterns]
        for pattern in pattern_mols:
            if mol_sample.HasSubstructMatch(pattern):
                return True
        return False
    elif monomer_class == "isocyanates":
        pattern = Chem.MolFromSmarts('[*]N=C=O')
        return mol_sample.HasSubstructMatch(pattern)
    else:
        raise ValueError("Invalid monomer class")

def evaluate_mol(smiles_samples, monomer_class):
    mol_samples = [get_mol(s) for s in smiles_samples]
    mol_samples = [x for x in mol_samples if x is not None]

    val_metric = len(mol_samples) / len(smiles_samples)

    div = InternalDiversity()
    div_metric = div.get_diversity(mol_samples)

    orig_retro_metric = retro_sender(mol_samples)
    retro_metric = orig_retro_metric * val_metric
    # retro_metric = 0

    mem_metric = sum([eval_membership(s, monomer_class) for s in mol_samples]) / len(smiles_samples)
    return val_metric, div_metric, retro_metric, mem_metric

if __name__ == "__main__":
    project_name = "Rule-ICL"
    group_name = "mole-rule-icl"

    parse_args()
    random.seed(FLAGS.seed)

    # 1. setup logger
    config = vars(FLAGS)

    exp_name = "-".join([f"{k[:3]}_{v}" for k, v in sorted(config.items()) if k != "eval_only"])
    wandb.init(project=project_name, group=group_name, name=exp_name, config=config)
    log_dir = f"log/{group_name}/{exp_name}"
    setup_logger_file(logger, log_dir)
    wandb.run.log_code("./neural_lark")

    # 1.1 setup llm
    llm = setup_llm(FLAGS.engine)

    # 2. load dataset and parser
    examples = load_dataset(config)
    global_parser, global_grammar_str = load_parser(config, use_generic_grammar=FLAGS.use_generic_grammar)

    # 3. generate samples
    if not FLAGS.eval_only:
        prompt_template = prompt_template[FLAGS.prompt_template]
        if "{name}" in prompt_template["instruction"]:
            prompt_template["instruction"] = prompt_template["instruction"].format_map({"name": config["dataset"]})
        num_samples = config["num_samples"]
        if FLAGS.prompt_mode == "rot":
            prompt_template["instruction"] = f"{prompt_template['instruction']}\n[BEGIN RULES]\n{global_grammar_str}[END RULES]\n\n"
            samples = batch_prompt_predict_with_rule(llm, examples, prompt_template, num_samples, constrain_prog=FLAGS.constrain_prog_gen_flag)
        else:
            samples = batch_prompt_predict(llm, examples, prompt_template, num_samples)
        
        # 4. log samples and evaluate
        with open(f"{log_dir}/samples.txt", "w") as f:
            for sample in samples:
                f.write(sample + "\n")

    else:
        sample_file = f"{log_dir}/samples.txt"
        logger.info(f"evaluating samples from {sample_file}")
        with open(sample_file, "r") as f:
            samples = [line.strip() for line in f.readlines()]
        
        valid_metric, div_metric, retro_metric, mem_metric = evaluate_mol(samples, monomer_class=config["dataset"])
        logger.info(f"validity: {valid_metric}, diversity: {div_metric}, retro: {retro_metric}, membership: {mem_metric}")

        # clear log files for retro star
        default_int_file = "log/deg/generated_samples.txt"
        with open(default_int_file, 'w') as fw:
            fw.write('')

        default_out_file = "log/deg/output_syn.txt"
        output_file = f"{log_dir}/output_syn.txt"
        shutil.copyfile(default_out_file, output_file)
        with open(default_out_file, 'w') as fw:
            fw.write('')
        
