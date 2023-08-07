import os
import json
import random
import tqdm

import wandb
import functools
import collections

import numpy as np
from minEarley.parser import EarleyParser

from neural_lark.flags import FLAGS, parse_args
from neural_lark.dataset import load_sempar_data, load_sem_parser, evaluate_programs, evaluate_grammars, counter2pred, evaluate_dfa, evaluate_fol
from neural_lark.llm_interface import setup_llm
from neural_lark.retriever import retrieve_fn_dict, setup_bm25
from neural_lark.earley import predict_program_with_earley_correction, predict_rules_with_earley_correction
from neural_lark.train_utils import logger, setup_logger_file
from neural_lark.lark_utils import * 
from neural_lark.overnight_utils import remove_lf_space as remove_lf_space_overnight


def construct_rule_instruction(rules, dataset):
    if dataset == "geoquery" or dataset == "overnight":
        instruction = "First, you should write grammar rules by choosing from the following BNF rules. Then, you should write programs that conform to your predicted rules.\n"
        add_rules_flag = True
    elif dataset == "smc" or dataset == "regex":
        instruction= "First, you should write a grammar that contains all the necessary BNF rules. Then, you should write programs that conform to your predicted rules.\n"
        add_rules_flag = False
    elif dataset == "folio":
        instruction = "First, you should write a BNF grammar that covers all the necessary predicates, constants and logical rules. Then, you should write first-order logic formulas that conform to your predicted rules. Note that constants should start with lowercase; predicates should start with uppercase.\n"
        add_rules_flag = True
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if add_rules_flag:
        lark_str = rulelist2larkstr(rules)
        bnf_str = lark2bnf(lark_str)
        instruction = f"{instruction}\n[BEGIN RULES]\n{bnf_str}\n[END RULES]\n\n" 
    return instruction

DELIMITER = "\nprogram based on the BNF grammar rules:\n"
prompt_templates = {
    "std": {
        "instruction": ("You are an expert programmer, and you need to write a program" 
                        " for the given natural language query.\n"),
        "rule_instruction": None,
        "exemplar": lambda ex: f"query: {ex.source}\nprogram:\n{ex.target}\n\n",
        "prediction": lambda ex: f"query: {ex.source}\nprogram:\n",
    },
    "wrule": {
        "instruction": ("You are an expert programmer, and you need to write a program" 
                        " for the given natural language query.\n"),
        "rule_instruction": "",
        "exemplar": lambda ex: f"query: {ex.source}\nBNF grammar rules:\n{ex.grammar}{DELIMITER}{ex.target}\n\n",
        "rule_exemplar": lambda ex: f"query: {ex.source}\nBNF grammar rules:\n{ex.grammar}\n\n",
        "prediction": lambda ex: f"query: {ex.source}\nBNF grammar rules:\n",
        "prediction_given_rule": lambda ex: f"query: {ex.source}\nBNF grammar rules:\n{ex.grammar}{DELIMITER}",
    },
    "fol_std": {
        "instruction": ("You are an expert programmer, and you need to write first-order logic formulas"
                        " for the given natural language sentences. The goal is to determine whether the final sentence can be inferred from the previous sentences.\n"),
        "rule_instruction": None,
        "exemplar": lambda ex: f"sentences:\n{ex.source}\nprogram:\n{ex.target}\n\n",
        "prediction": lambda ex: f"sentences:\n{ex.source}\nprogram:\n",
    },
    "fol_wrule": {
        "instruction": ("You are an expert programmer, and you need to write first-order logic formulas"
                        " for the given natural language sentences. The goal is to determine whether the final sentence can be inferred from the previous sentences.\n"),
        "rule_instruction": "",
        "exemplar": lambda ex: f"sentences:\n{ex.source}\nBNF grammar rules:\n{ex.grammar}{DELIMITER}{ex.target}\n\n",
        "rule_exemplar": lambda ex: f"sentences:\n{ex.source}\nBNF grammar rules:\n{ex.grammar}\n\n",
        "prediction": lambda ex: f"sentences:\n{ex.source}\nBNF grammar rules:\n",
        "prediction_given_rule": lambda ex: f"sentences:\n{ex.source}\nBNF grammar rules:\n{ex.grammar}{DELIMITER}",
    }
}

def batch_prompt_predict(
    llm,
    input_examples, 
    train_examples, 
    prompt_template, 
    retrieve_fn,
    use_linearized_tree_flag,
    constrain_prog_gen_flag,
    overnight_flag,
    predefined_fewshot_prompt=None,
):
    """
    Args:
        overnight_flag: overnight lf needs special handling when using linearized tree
        fewshot_prompt: if None, construct the prompt from the template
    """
    if use_linearized_tree_flag:
        assert not constrain_prog_gen_flag, "linearized tree is not compatible with earley correction"

    prompts, predictions = [], []
    template_ex = prompt_template["exemplar"]
    template_p = prompt_template["prediction"]
    for input_example in tqdm.tqdm(input_examples, total=len(input_examples)):

        if predefined_fewshot_prompt is None:
            fewshot_prompt = prompt_template["instruction"]
            if prompt_template["rule_instruction"]:
                fewshot_prompt += prompt_template["rule_instruction"]
            exemplars = retrieve_fn(input_example, train_examples)
            for exemplar in exemplars:
                if use_linearized_tree_flag:
                    if not hasattr(exemplar, "linearized"):
                        exemplar_tree = global_parser.parse(exemplar.target)
                        exemplar.target = linearize_tree(exemplar_tree)
                        exemplar.linearized = True
                fewshot_prompt += template_ex(exemplar)
        else:
            fewshot_prompt = predefined_fewshot_prompt

        _prompt = fewshot_prompt + template_p(input_example)
        prompts.append([_prompt])

        ret_predictions = []
        if constrain_prog_gen_flag:
            prediction = predict_program_with_earley_correction(llm, _prompt, global_parser)
            ret_predictions.append(prediction)
        else:
            responses = llm.sample_completions(_prompt, FLAGS.temperature, stop_token="\n\n")
            assert len(responses) == 1
            prediction = responses[0].response_text

            if use_linearized_tree_flag:
                # recover the original program
                logger.debug("prediction before linearization: " + prediction)
                if overnight_flag:
                    prediction = linearized_tree_to_program(prediction, delimiter=" ")
                    prediction = remove_lf_space_overnight(prediction)
                else:
                    prediction = linearized_tree_to_program(prediction)

            ret_predictions.append(prediction)
                    
        _counter = collections.Counter(ret_predictions)
        predictions.append(_counter)
        logger.info("Summary:" + "-" * 80)
        logger.info(f"number of unique predictions from std prompt: {len(_counter)}")
        logger.info(f"frequency distribution of new predictions: {list(_counter.values())}")

        logger.info(f"    source:\n{input_example.source}")
        logger.info(f"prediction:\n{counter2pred(_counter)}")
        logger.info(f"    target:\n{input_example.target}")
        logger.info("-" * 80)
    return prompts, predictions

def batch_prompt_wrule_predict(
        llm,
        input_examples,
        train_examples,
        prompt_template,
        retrieve_fn,
        use_oracle_rule_flag,
        constrain_rule_gen_flag,
        constrain_prog_gen_flag,
        separate_rule_gen_flag,
        lazy_constrain_flag,
        predefined_fewshot_prompt=None,
    ):
    """
    Args:
        use_oracle_rule_flag: if True, use oracle rule to generate the prompt
        constrain_rule_gen_flag: if True, constrain rule generation
        constrain_prog_gen_flag: if True, constrain program generation
        seperate_rule_gen_flag: if True, generate rule first, then program using different prompts
        lazy_constrain_flag: sample k candidates first; if no candidate is valid, then use early two-stage generation
    """
    prompts, predictions, grammars = [], [], []
    template_rule_prog_ex = prompt_template["exemplar"]
    template_rule_ex = prompt_template["rule_exemplar"]
    template_starts_wrule_pred = prompt_template["prediction"]
    template_prog_given_rule_pred = prompt_template["prediction_given_rule"]

    for input_example in tqdm.tqdm(input_examples, total=len(input_examples)):
        if predefined_fewshot_prompt is None:
            exemplars = retrieve_fn(input_example, train_examples)
            fewshot_rule_prog_prompt = prompt_template["instruction"]
            if prompt_template["rule_instruction"]:
                fewshot_rule_prog_prompt += prompt_template["rule_instruction"]
            for exemplar in exemplars:
                exemplar.grammar = lark2bnf(gen_min_lark(exemplar.target, global_parser))
                fewshot_rule_prog_prompt += template_rule_prog_ex(exemplar)
        else:
            fewshot_rule_prog_prompt = predefined_fewshot_prompt
        
        do_earley_two_stage_gen_flag = False
        if use_oracle_rule_flag:
            bnf_grammar = lark2bnf(gen_min_lark(input_example.target, global_parser))
            input_example.grammar = bnf_grammar
            prompt_for_prog = fewshot_rule_prog_prompt + template_prog_given_rule_pred(input_example)

            lark_grammar = bnf2lark(bnf_grammar)
            assert check_grammar_validity(global_rules, lark_grammar)
            ret_grammars = [lark_grammar]

            if constrain_prog_gen_flag:
                prediction = predict_program_with_earley_correction(llm, prompt_for_prog, global_parser)
                ret_predictions = [prediction]
            else:
                response = llm.sample_completions(prompt_for_prog, FLAGS.temperature, stop_token="\n\n")[0]
                ret_predictions = [response.response_text] 
        elif lazy_constrain_flag:
            assert not separate_rule_gen_flag
            prompt_for_rule_prog = fewshot_rule_prog_prompt + template_starts_wrule_pred(input_example)
            responses = llm.sample_completions(prompt_for_rule_prog, FLAGS.temperature, stop_token="\n\n")
            raw_predictions = [r.response_text for r in responses] 
            
            ret_predictions, ret_grammars = [], []
            for raw_pred in raw_predictions:
                try:
                    pred_bnf_grammar, pred_program = raw_pred.split(DELIMITER)
                    pred_lark_grammar = bnf2lark(pred_bnf_grammar)
                    if constrain_rule_gen_flag and not check_grammar_validity(global_rules, pred_lark_grammar):
                        continue

                    if constrain_prog_gen_flag:
                        local_parser = EarleyParser(decorate_grammar(pred_lark_grammar), start=global_parser.option.start)
                        local_parser.parse(pred_program)

                    ret_grammars.append(pred_lark_grammar)
                    ret_predictions.append(pred_program)
                except Exception as e:
                    logger.warning(f"failed to find prediction from {raw_pred} due to {e}")

            if len(ret_predictions) == 0:
                logger.info("invoking earley correction")
                do_earley_two_stage_gen_flag = True
        else:
            assert not separate_rule_gen_flag
            do_earley_two_stage_gen_flag = True

        if do_earley_two_stage_gen_flag:
            # if separate, use another prompt for rule generation
            if separate_rule_gen_flag:
                fewshot_rule_prompt = prompt_template["instruction"]
                if prompt_template["rule_instruction"]:
                    fewshot_rule_prompt += prompt_template["rule_instruction"]
                for exemplar in exemplars:
                    exemplar.grammar = lark2bnf(gen_min_lark(exemplar.target, global_parser))
                    fewshot_rule_prompt += template_rule_ex(exemplar)
                prompt_for_rule = fewshot_rule_prompt + template_starts_wrule_pred(input_example)
            else:
                prompt_for_rule = fewshot_rule_prog_prompt + template_starts_wrule_pred(input_example)

            try:
                if constrain_rule_gen_flag:
                    pred_bnf_grammar = predict_rules_with_earley_correction(llm, prompt_for_rule, global_rules, DELIMITER)
                else:
                    response = llm.sample_completions(prompt_for_rule, FLAGS.temperature, stop_token=DELIMITER)[0]
                    pred_bnf_grammar = response.response_text 
                pred_lark_grammar = bnf2lark(pred_bnf_grammar)
                input_example.grammar = pred_bnf_grammar
                prompt_for_prog = fewshot_rule_prog_prompt + template_prog_given_rule_pred(input_example)

                if constrain_prog_gen_flag:
                    try:
                        logger.info(f"earley correction with grammar\n{pred_lark_grammar}")
                        local_parser = EarleyParser(decorate_grammar(pred_lark_grammar), start=global_parser.option.start)
                    except Exception as e:
                        logger.warning(f"failed to create parser due to {e}, reverting to global parser")
                        local_parser = global_parser
                    pred_program = predict_program_with_earley_correction(llm, prompt_for_prog, local_parser)
                else:
                    resposne = llm.sample_completions(prompt_for_prog, FLAGS.temperature, stop_token="\n\n")[0]
                    pred_program = resposne.response_text 
            except Exception as e:
                logger.warning(f"failed to find prediction due to {e}")
                prompt_for_rule_prog = fewshot_rule_prog_prompt + template_starts_wrule_pred(input_example) 
                response = llm.sample_completions(prompt_for_rule_prog, FLAGS.temperature, stop_token="\n\n")[0]
                try:
                    pred_bnf_grammar, pred_program = response.split(DELIMITER)
                    pred_lark_grammar = bnf2lark(pred_bnf_grammar)
                except:
                    logger.warning(f"failed to find prediction from {response.response_text} due to {e}")
                    pred_lark_grammar, pred_program = None, None
            
            ret_grammars = [pred_lark_grammar]
            ret_predictions = [pred_program]

        # collect prompts and predictions
        used_prompts = []
        if "prompt_for_prog" in locals():
            used_prompts.append(prompt_for_prog)
        if "prompt_for_rule_prog" in locals():
            used_prompts.append(prompt_for_rule_prog)
        if "prompt_for_rule" in locals():
            used_prompts.append(prompt_for_rule)
        prompts.append(used_prompts)
        _pred_counter = collections.Counter(ret_predictions)
        predictions.append(_pred_counter)
        _grammar_counter = collections.Counter(ret_grammars)
        grammars.append(_grammar_counter)

        logger.info("Summary:" + "-" * 80)
        logger.info(f"number of unique predictions: {len(_pred_counter)}")
        logger.info(f"frequency distribution of predictions: {list(_pred_counter.values())}")

        logger.info(f"    source:\n{input_example.source}")
        logger.info(f"prediction:\n{counter2pred(_pred_counter)}")
        logger.info(f"    target:\n{input_example.target}")
        logger.info(f"   grammar:\n{counter2pred(_grammar_counter)}")
        logger.info("-" * 80)
    return prompts, predictions, grammars


if __name__ == "__main__":
    # 0. meta info
    project_name = "Rule-ICL"
    group_name = "sempar-rule-icl"

    # 1. setup 
    ##1.1 wandb and logger
    parse_args()
    random.seed(FLAGS.seed)
    config = vars(FLAGS)
    exp_name = "-".join([f"{k[:3]}_{v}" for k, v in sorted(config.items()) if k not in ["eval_only"]])
    wandb.init(project=project_name, group=group_name, name=exp_name, config=config)
    log_dir = f"log/{group_name}/{exp_name}"
    setup_logger_file(logger, log_dir)
    wandb.run.log_code("./neural_lark")

    ##1.2 setup grammar and parser
    global_parser, global_rules = load_sem_parser(config)

    ## 1.3 setup llm
    llm = setup_llm(FLAGS.engine)

    # 2. load data
    train_examples, dev_examples, test_examples = load_sempar_data(config)
    logger.info(f"loaded {len(train_examples)} indist examples, {len(dev_examples)} dev examples, {len(test_examples)} test examples")

    #3. prepare retrieval func and prompt template
    retrieve_fn = retrieve_fn_dict[FLAGS.retrieve_fn]
    if config["retrieve_fn"] == "bm25":
        bm25 = setup_bm25(train_examples)
        retrieve_fn = functools.partial(retrieve_fn, batch_size=FLAGS.batch_size, bm25=bm25)
    else:
        retrieve_fn = functools.partial(retrieve_fn, batch_size=FLAGS.batch_size)

    prompt_template = prompt_templates[FLAGS.prompt_template]
    if FLAGS.add_rule_instruction_flag:
        new_instruction = construct_rule_instruction(global_rules, FLAGS.dataset)
        prompt_template["rule_instruction"] = new_instruction
    
    # 4. few-shot prompting
    assert FLAGS.prompt_mode in ["std", "rot"]
    logger.info("few-shot prompting on the test set")
    if not config["eval_only"]:
        if FLAGS.prompt_from_file:
            with open(FLAGS.prompt_from_file) as f:
                predefined_fewshot_prompt = f.read()
        else:
            predefined_fewshot_prompt = None

        if  config["prompt_mode"] == "std":
            test_prompts, test_prediction_counters = batch_prompt_predict(llm, test_examples, train_examples, prompt_template, retrieve_fn, use_linearized_tree_flag=FLAGS.use_linearized_tree, constrain_prog_gen_flag=FLAGS.constrain_prog_gen_flag, overnight_flag=FLAGS.dataset=="overnight", predefined_fewshot_prompt=predefined_fewshot_prompt)
            test_grammar_counters = None
        else:
            test_prompts, test_prediction_counters, test_grammar_counters = batch_prompt_wrule_predict(llm, test_examples, train_examples, prompt_template, retrieve_fn, use_oracle_rule_flag=FLAGS.use_oracle_rule_flag, separate_rule_gen_flag=FLAGS.separate_rule_gen_flag, constrain_rule_gen_flag=FLAGS.constrain_rule_gen_flag, constrain_prog_gen_flag=FLAGS.constrain_prog_gen_flag, lazy_constrain_flag=FLAGS.lazy_constrain_flag, predefined_fewshot_prompt=predefined_fewshot_prompt)

        ##  dump to json and wandb
        json_results = {
            "test_prompts": test_prompts,
            "test_predictions": test_prediction_counters,
            "test_grammars": test_grammar_counters,
        }

        with open(f"{log_dir}/results.json", "w") as f:
            logger.info(f"dumping results to {log_dir}/results.json")
            json.dump(json_results, f, indent=2)

    else:
        # load from json
        with open(f"{log_dir}/results.json", "r") as f:
            json_results = json.load(f)
        test_prediction_counters = [collections.Counter(d) for d in json_results["test_predictions"]]
        test_grammar_counters = [collections.Counter(d) for d in json_results["test_grammars"]]

    ## 4.1 evaluation
    if config["dataset"] == "regex":
        test_accuracy = evaluate_dfa(test_prediction_counters, test_examples)
        test_grammar_accuracy = 0.0
    elif config["dataset"] == "folio":
        test_accuracy = evaluate_fol(test_prediction_counters, test_examples, global_parser)
        test_grammar_accuracy = 0.0
    else:
        test_accuracy = evaluate_programs(test_prediction_counters, test_examples)
        # test_grammar_accuracy = evaluate_grammars(test_grammar_counters, test_examples, global_parser)
        test_grammar_accuracy = 0.0
        logger.info(f"test accuracy {test_accuracy}")

    ## log to wandb
    wandb.log({
        "test_accuracy": test_accuracy, 
        "test_grammar_accuracy": test_grammar_accuracy
    }, step=1)