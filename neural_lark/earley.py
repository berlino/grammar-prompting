import collections
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModel

from neural_lark.train_utils import logger
from neural_lark.lark_utils import *

sb_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2')
sb_model = AutoModel.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2')
sb_model.eval()

def score_by_sentencebert(prediction, candidate):
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    encoded_input = sb_tokenizer([prediction, candidate], padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = sb_model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        score = torch.cosine_similarity(sentence_embeddings[0], sentence_embeddings[1], dim=0)
        return score.item()


def predict_program_with_earley_correction(llm, prompt, parser):
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
        response = llm.greedy_completion(_prompt, stop_token="\n\n")
        residual_program_prediction = response.response_text 

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
        for pair_idx, pair in enumerate(pairs):
            logger.debug(f"candidate {pair_idx}: prefix [{pair[0]}] suffix [{pair[1]}]")
        logger.info(f"number of candidates: {len(pairs)}")
        scores = []
        for prefix, suffix in pairs:

            # no longer supported due to API change
            # _prompt = prompt + prefix
            # score = llm.evaluate_completion(_prompt, suffix, average=True)

            candidate = prefix + suffix
            score = score_by_sentencebert(program_prediction, candidate) 
            scores.append(score)

        best_idx = scores.index(max(scores))
        fixed_prediction = pairs[best_idx][0] + pairs[best_idx][1]
        logger.warning(f"fixed prediction: {fixed_prediction}")

        partial_program_prediction = fixed_prediction
        num_correction_left -= 1
    
    if ret_prediction is None:
        logger.warning(f"cannot find a valid prediction after {MAX_NUM_CORRECTION} retries")
        ret_prediction = initial_prediction
    
    return ret_prediction

def predict_rules_with_earley_correction(llm, prompt, ruleset, delimiter):
    """
    Predict grammar rules with earley correction.
    Args:
        delimiter: the separator between rule and program
    """
    MAX_NUM_CORRECTION = 3
    CANDIDATE_NUM_THRESHOLD = 16
    num_correction_left = MAX_NUM_CORRECTION

    rules_by_origin = collections.defaultdict(list)
    for rule in ruleset:
        rules_by_origin[rule.origin].append(rule)
    
    def validate_rule(prediction):
        pred_lark = bnf2lark(prediction)
        pred_rulelist = larkstr2rulelist(pred_lark)  # an ordered list of rules
        for pred_rule in pred_rulelist:
            if pred_rule not in ruleset and pred_rule.origin not in skipped_nonterminal_names:
                logger.debug(f"found an invalid rule: {pred_rule}")
                return False
        return True
    
    def filter_candidates(pred_rule, candidates):
        if len(candidates) > CANDIDATE_NUM_THRESHOLD:
            scores = []
            for candidate in candidates:

                pred_rulename = str(pred_rule)
                score = score_by_sentencebert(pred_rulename, str(candidate))
                scores.append(score)
            top_candidates = [candidates[i] for i in np.argsort(scores)[-CANDIDATE_NUM_THRESHOLD:]]
            candidates = top_candidates
        
        return candidates 
    
    def obtain_correction_pairs(prediction):
        """
        Returns a list of candidates in the form of (prefix, suffix).
        """
        pred_lark = bnf2lark(prediction)
        pred_rulelist = larkstr2rulelist(pred_lark)  # an ordered list of rules

        lhs_set = set()
        partial_rule_list = []
        for pred_rule in pred_rulelist:
            if pred_rule not in ruleset and pred_rule.origin not in skipped_nonterminal_names:
                # find condidates considering the origin of the rule
                if pred_rule.origin in lhs_set:
                    candidates = [r for r in rules_by_origin[pred_rule.origin] if r not in partial_rule_list]
                else:
                    candidates = [r for r in ruleset if r not in partial_rule_list]
                candidates = filter_candidates(pred_rule, candidates)

                logger.info(f"number of candidates for correction: {len(candidates)}")
                for candidate_idx, candidate in enumerate(candidates):
                    logger.debug(f"candidate {candidate_idx}: [{candidate}]")

                # serialize the partial rule list
                ret_pairs = []
                prefix = rulelist2bnfstr(partial_rule_list)
                for candidate in candidates:
                    first_rhs = candidate.origin not in lhs_set
                    if first_rhs:
                        suffix = "\n" + candidate.to_bnf()
                    else:
                        suffix = " | " + ' '.join(candidate.expansion)
                    ret_pairs.append((prefix, suffix))
                
                return ret_pairs
            else:
                # avoid duplicate rules
                if pred_rule not in partial_rule_list:
                    if pred_rule.origin not in lhs_set:
                        lhs_set.add(pred_rule.origin)
                    partial_rule_list.append(pred_rule)
    
    partial_rule_prediction = ""
    ret_prediction, initial_prediction = None, None

    while num_correction_left > 0:
        _prompt = prompt + partial_rule_prediction
        response = llm.greedy_completion(_prompt, stop_token="\n\n") 
        residual_rule_prediction = response.response_text.split(delimiter)[0]

        if initial_prediction is None:
            initial_prediction = residual_rule_prediction
        rule_prediction = partial_rule_prediction + residual_rule_prediction
        logger.debug(f"partial rule prediction: {rule_prediction}")
        
        if validate_rule(rule_prediction):
            ret_prediction = rule_prediction
            break

        pairs = obtain_correction_pairs(rule_prediction)
        assert len(pairs) > 0, "no correction pairs found"
        logger.debug(f"number of candidates: {len(pairs)}")

        scores = []
        for prefix, suffix in pairs:
            # no longer supported due to API change
            # _prompt = prompt + prefix
            # score = llm.evaluate_completion(_prompt, suffix, average=True)

            candidate = prefix + suffix
            score = score_by_sentencebert(rule_prediction, candidate)

            scores.append(score)
        best_idx = scores.index(max(scores))
        fixed_rule_prediction = pairs[best_idx][0] + pairs[best_idx][1]
        logger.debug(f"fixed rule: {pairs[best_idx][1]}")
        logger.debug(f"fixed partial rule prediction:\n{fixed_rule_prediction}")

        partial_rule_prediction = fixed_rule_prediction
        num_correction_left -= 1
        
    if ret_prediction is None:
        logger.warning(f"cannot find a valid rule prediction after {MAX_NUM_CORRECTION} retries")
        ret_prediction = initial_prediction
    
    return ret_prediction
