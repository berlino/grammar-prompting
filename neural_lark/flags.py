import argparse

FLAGS = argparse.Namespace()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split_name", type=str)
    parser.add_argument("--domain", type=str)
    parser.add_argument("--num_shot", type=int)
    parser.add_argument("--dryrun", action="store_true")
    parser.add_argument("--quickrun", action="store_true")
    parser.add_argument("--eval_only", action="store_true")

    # llm
    parser.add_argument("--engine", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--freq_penalty", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--llm_cache_dir", type=str, default="llm_cache")

    # for generation
    parser.add_argument("--use_generic_grammar", action="store_true")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--rule_temperature", type=float, default=0.0)

    # prompting
    parser.add_argument("--prompt_mode", type=str, required=True)
    parser.add_argument("--prompt_template", type=str, required=True)
    parser.add_argument("--add_rule_instruction_flag", action="store_true")
    parser.add_argument("--retrieve_fn", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--prompt_from_file", type=str, default=None)

    ## for std prompting
    parser.add_argument("--use_linearized_tree", action="store_true")

    ## for grammar-based prompting
    parser.add_argument("--use_oracle_rule_flag", action="store_true")
    parser.add_argument("--separate_rule_gen_flag", action="store_true")
    parser.add_argument("--constrain_rule_gen_flag", action="store_true")
    parser.add_argument("--constrain_prog_gen_flag", action="store_true")
    parser.add_argument("--lazy_constrain_flag", action="store_true")

    ## for iterative prompting
    parser.add_argument("--num_iterations", type=int, default=1)
    parser.add_argument("--iter_prompt_template", type=str, default="iter")
    parser.add_argument("--iter_engine", type=str)
    parser.add_argument("--iter_retrieve_fn", type=str)

    args = parser.parse_args()
    FLAGS.__dict__.update(args.__dict__)