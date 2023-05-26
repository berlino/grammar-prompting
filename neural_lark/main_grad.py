import os
import json
import random
import tqdm

import wandb
import functools
import collections
import pandas as pd
from dataclasses import dataclass

from neural_lark.flags import FLAGS, parse_args
from neural_lark.dataset import load_sempar_data, load_sem_parser, evaluate_programs, counter2pred, evaluate_fol
from neural_lark.llm_interface import setup_llm
from neural_lark.retriever import retrieve_fn_dict, setup_bm25
from neural_lark.train_utils import logger, setup_logger_file


prompt_templates = {
    "std": {
        "instruction": "You are an expert programmer, and you need to write a program for the given natural language query.\n",
        "exemplar": lambda ex: f"query:\n{ex.source}\nprogram: {ex.target}\n\n",
        "prediction": lambda ex: f"query:\n{ex.source}\nprogram: ",
    },

    "refine": {
        "instruction": ("You are an expert programmer, and you need write a program"
                        " for the given natural language query by copying or improving the base program.\n"),
        "exemplar": lambda ex: f"query:\n{ex.source}\nbase program:\n{ex.prediction}\nreturn program:\n{ex.target}\n\n",
        "prediction": lambda ex: f"query:\n{ex.source}\nbase program:\n{ex.prediction}\nreturn program:\n",
    },
}

def batch_init_prompt_predict(llm, input_examples, train_examples, prompt_template, retrieve_fn):
    prompts, predictions = [], []
    template_ex = prompt_template["exemplar"]
    template_p = prompt_template["prediction"]
    for input_example in tqdm.tqdm(input_examples, total=len(input_examples)):
        fewshot_prompt = prompt_template["instruction"]
        exemplars = retrieve_fn(input_example, train_examples)
        for exemplar in exemplars:
            fewshot_prompt += template_ex(exemplar)

        _prompt = fewshot_prompt + template_p(input_example)
        prompts.append([_prompt])

        responses = llm.sample_completions(_prompt, FLAGS.temperature, stop_token="\n\n")
        new_predictions = [r.response_text for r in responses]
        _counter = collections.Counter(new_predictions)
        predictions.append(_counter)

        logger.info(f"    source:\n{input_example.source}")
        logger.info(f"prediction:\n{counter2pred(_counter)}")
        logger.info(f"    target:\n{input_example.target}")
        logger.info("-" * 60)
    return prompts, predictions

def batch_iter_prompt_predict(llm, input_examples, input_predictions, train_examples, prompt_template, retrieve_fn):
    """
    Args:
        input_predictions: list of list of predictions
    
    Returns:
        prompts: list of list of prompts
        predictions: list of list of new predictions
    """
    prompts, predictions = [], []
    template_ex = prompt_template["exemplar"]
    template_p = prompt_template["prediction"]
    for input_example, input_prev_predictions_counter in tqdm.tqdm(zip(input_examples, input_predictions), total=len(input_examples)):
        input_example.prediction = counter2pred(input_prev_predictions_counter)

        exemplars = retrieve_fn(input_example, train_examples)
        fewshot_prompt = prompt_template["instruction"]
        for exemplar in exemplars:
            fewshot_prompt += template_ex(exemplar)
        _prompt = fewshot_prompt + template_p(input_example)
        prompts.append(_prompt)

        responses = llm.sample_completions(_prompt, FLAGS.temperature, stop_token="\n\n") 
        new_predictions = [r.response_text for r in responses]
        _counter = collections.Counter(new_predictions)
        predictions.append(_counter)

        logger.info(f"        source:\n{input_example.source}")
        logger.info(f"old prediction:\n{input_example.prediction}")
        logger.info(f"new prediction:\n{counter2pred(_counter)}")
        logger.info(f"        target:\n{input_example.target}")
        logger.info("-" * 60)
    return prompts, predictions

def prepare_retr_fn(retr_fn_name, retr_examples):
    retrieve_fn = retrieve_fn_dict[retr_fn_name]
    if retr_fn_name == "bm25":
        bm25_init = setup_bm25(retr_examples)
        retrieve_fn = functools.partial(retrieve_fn, batch_size=FLAGS.batch_size, bm25=bm25_init)
    else:
        retrieve_fn = functools.partial(retrieve_fn, batch_size=FLAGS.batch_size)
    return retrieve_fn

def evaluate(dataset, prediction_counters, examples, parser=None):
    if dataset == "folio":
        accuracy = evaluate_fol(prediction_counters, examples, parser)
    else:
        accuracy = evaluate_programs(prediction_counters, examples)
    return accuracy

if __name__ == "__main__":
    # 0. meta info
    project_name = "CorrICT-LLM"
    group_name = "sempar-sym_grad-icl"

    # 1. setup
    parse_args()
    random.seed(FLAGS.seed)
    config = vars(FLAGS)
    exp_name = "-".join([f"{k[:3]}_{v}" for k, v in sorted(config.items()) if k not in ["eval_only"]])
    wandb.init(project=project_name, group=group_name, name=exp_name, config=config)
    log_dir = f"log/{group_name}/{exp_name}"
    setup_logger_file(logger, log_dir)
    wandb.run.log_code("./neural_lark")

    ## 1.2 setup llm
    init_llm = setup_llm(FLAGS.engine)
    iter_llm = setup_llm(FLAGS.iter_engine)
    
    # 2. load data
    train_examples, dev_examples, test_examples = load_sempar_data(config)
    # split_point = int(len(train_examples) * 0.5)
    # train_examples4init_icl, train_examples4iter_icl = train_examples[:split_point], train_examples[split_point:]
    train_examples4init_icl, train_examples4iter_icl = train_examples, train_examples
    parser, _ = load_sem_parser(config)
    
    logger.info(f"loaded {len(train_examples4init_icl)} indist examples, {len(train_examples4iter_icl)} oodist examples, {len(dev_examples)} dev examples, {len(test_examples)} test examples")

    #3. init prompting
    init_retrieve_fn = prepare_retr_fn(FLAGS.retrieve_fn, train_examples4init_icl)

    init_prompt_template = prompt_templates[FLAGS.prompt_template]
    summary = {"train_accuracy": [], "test_accuracy": []}
    
    logger.info("init few-shot prompting on the training set")
    train_prompts, train_prediction_counters = batch_init_prompt_predict(init_llm, train_examples4iter_icl, train_examples4init_icl, init_prompt_template, init_retrieve_fn)

    train_accuracy = evaluate(config["dataset"], train_prediction_counters, train_examples4iter_icl, parser=parser)
    summary["train_accuracy"].append(train_accuracy)
    logger.info(f"init train accuracy {train_accuracy}")

    logger.info("init few-shot prompting on the test set")
    test_prompts, test_prediction_counters = batch_init_prompt_predict(init_llm, test_examples, train_examples4init_icl, init_prompt_template, init_retrieve_fn)
    test_accuracy = evaluate(config["dataset"], test_prediction_counters, test_examples, parser=parser)
    summary["test_accuracy"].append(test_accuracy)
    logger.info(f"init test accuracy {test_accuracy}")

    ## dump to json
    init_json_results = {
        "train_prompts": train_prompts,
        "test_prompts": test_prompts,
        "train_predictions": train_prediction_counters,
        "test_predictions": test_prediction_counters,
    }
    with open(f"{log_dir}/std_prompt_results.json", "w") as f:
        json.dump(init_json_results, f, indent=2)

    ## log to wandb
    wandb.log({
        "train_prompts": wandb.Table(data=train_prompts, columns=["prompt"]),
        "test_prompts": wandb.Table(data=test_prompts, columns=["prompt"]),
    }, step=1)
    wandb.log({"train_accuracy": train_accuracy, "test_accuracy": test_accuracy}, step=1)

    #4. iterative correction-based prompting
    iter_prompt_template = prompt_templates[FLAGS.iter_prompt_template]
    iter_retrieve_fn = prepare_retr_fn(FLAGS.iter_retrieve_fn, train_examples4iter_icl)

    iter_train_examples = train_examples4iter_icl
    iter_num_steps = FLAGS.num_iterations
    assert iter_num_steps <= 1
    for step in range(2, iter_num_steps + 2):
        logger.info(f"start step {step}")

        ## 4.1 assign candidates
        for t_example, t_prediction_counter in zip(train_examples4iter_icl, train_prediction_counters):
            t_example.prediction = counter2pred(t_prediction_counter)

        ## 4.2 run ICL
        # logger.info("obtaininig predictions on the training set")
        # new_train_prompts, new_train_prediction_counters = batch_iter_prompt_predict(iter_llm, train_examples4iter_icl, train_prediction_counters, iter_train_examples, iter_prompt_template, iter_retrieve_fn)

        # train_accuracy = evaluate(config["dataset"], new_train_prediction_counters, train_examples4iter_icl, parser=parser)
        # summary["train_accuracy"].append(train_accuracy)
        # logger.info(f"finishes step {step}, train accuracy: {train_accuracy}")

        logger.info("obtaining predictions on the test set")
        new_test_prompts, new_test_prediction_counters = batch_iter_prompt_predict(iter_llm, test_examples, test_prediction_counters, iter_train_examples, iter_prompt_template, iter_retrieve_fn)

        test_accuracy = evaluate(config["dataset"], new_test_prediction_counters, test_examples, parser=parser)
        summary["test_accuracy"].append(test_accuracy)
        logger.info(f"finishes step {step}, test accuracy: {test_accuracy}")

        ## 4.5 dump to json and log to wandb
        iter_json_results = {
            # "train_prompts": new_train_prompts,
            "test_prompts": new_test_prompts,
            # "train_predictions": new_train_prediction_counters,
            "test_predictions": new_test_prediction_counters,
        }
        with open(f"{log_dir}/iter_prompt_results_step{step}.json", "w") as f:
            json.dump(iter_json_results, f, indent=2)

        wandb.log({"train_accuracy": train_accuracy, "test_accuracy": test_accuracy}, step=step)
        old_test_predictions = [counter2pred(counter)  for counter in test_prediction_counters]
        new_test_predictions = [counter2pred(counter)  for counter in new_test_prediction_counters]
        wandb.log(
            {
                "test_predictions": wandb.Table(
                    dataframe=pd.DataFrame(
                        {
                            "source": [e.source for e in test_examples],
                            "old_prediction": old_test_predictions,
                            "new_prediction": new_test_predictions,
                            "target": [e.target for e in test_examples],
                            "correction": [old_p != new_p for old_p, new_p in zip(old_test_predictions, new_test_predictions)],
                            "correctness": [p == e.target for p, e in zip(new_test_predictions, test_examples)],
                        }
                    )
                ),
            },
            step=step,
        )

        ## 4.6 update the predictions for another round of correction
        # train_prediction_counters = new_train_prediction_counters
        test_prediction_counters = new_test_prediction_counters
    
    logger.info(f"summary: {summary}")
