import itertools
import collections

import re
import lark
from lark.load_grammar import _TERMINAL_NAMES, load_grammar
from minEarley.tree import Tree

from dataclasses import dataclass
from neural_lark.train_utils import logger

from stanza.models.constituency.parse_tree import Tree as StanzaTree

"""
For convenince, we use SimpleRule instead of lark.grammar.Rule for 1) putting rules 
in the instruction, 2) check if model-generated rules are valid. 
In the future, we may want to directly use lark.grammar.Rule, e.g., let the model
generate rules in EBNF or BNF format.
"""

# these nonterminals will be inlined when constructing rules
inline_terminal_names = {
        # for SMC dataset
        "WORD", "NUMBER", "ESCAPED_STRING", "L", 
        # for regex dataset
        "STRING", "INT", "CHARACTER_CLASS", "CONST",
        # for overnight
        # "PROPERTY", "SINGLETON_VALUE", "ENTITY_VALUE", "NUMBER_VALUE",
        # for molecule
        "N", "C", "O", "F", "c",
        # for fol
        "PNAME", "CNAME", "LCASE_LETTER"
}
for k, v in _TERMINAL_NAMES.items():
    inline_terminal_names.add(v)

## these are the nonterminals that are not needed to be predicted from model, will be used to to check the validity of the generated rules
skipped_nonterminal_names = (
    # for smc and regex
    "string", "number", "literal", "delimiter",
    # "VALUE"  # for mtop
    # "property", "value",  # for overnight
)

"""
Some concepts:
    - larkstr: a string in Lark format 
    - bnfstr: a string in BNF format (use ::= instead of :)
"""


# poor man's rule
@dataclass
class SimpleRule:
    origin: str
    expansion: tuple

    def __hash__(self):
        return hash(str(self))
    
    def __str__(self):
        return self.to_lark()
    
    def to_lark(self):
        return f"{self.origin} : {' '.join(self.expansion)}"
    
    def to_bnf(self):
        return f"{self.origin} ::= {' '.join(self.expansion)}"
    
    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, SimpleRule):
            return False
        return str(self) == str(__o)

def _wrap_string(s):
    if s.startswith("\"") and s.endswith("\""):
        # a bit complex to preserve the quotation marks
        s = f"\"\\{s[:-1]}\\\"{s[-1]}"
    else:
        s = f"\"{s}\""    

    # escape unicode characters 
    if "\\u" in s:
        s = s.replace("\\u", "\\\\u")
    
    return s

def split_rule(rule):
    split_idx = rule.index(":")
    lhs, rhs = rule[:split_idx].strip(), rule[split_idx+1:].strip()
    return lhs, rhs

def treenode2rule(treenode):
    if treenode is None:
        return None

    if isinstance(treenode, Tree):
        origin = f"{treenode.data.value}"
        expansion = []

        for child in treenode.children:
            if child is None:
                continue

            if isinstance(child, Tree):
                expansion.append(child.data.value)
            else:
                if child.type.startswith("__") or child.type in inline_terminal_names:
                    expansion.append(_wrap_string(child.value))
                else:
                    expansion.append(child.type)
    else: # terminal
        if treenode.type.startswith("__") or treenode.type in inline_terminal_names:
            return None
        else:
            origin = treenode.type
            expansion = [_wrap_string(treenode.value)]
    return SimpleRule(origin, tuple(expansion))
    
def extract_rule_stat(tree, rule_stat):
    """
    Count the occurrence of each rule
    """
    cur_rule = treenode2rule(tree)
    if cur_rule is None:
        return
    if cur_rule not in rule_stat:
        rule_stat[cur_rule] = 1
    else:
        rule_stat[cur_rule] += 1

    if getattr(tree, "children", None):
        for child in tree.children:
            extract_rule_stat(child, rule_stat)

def tree2rulelist(tree):
    rule_list = []
    def recur_add(node, rule_list):
        cur_rule = treenode2rule(node)
        if cur_rule is None:
            return
        rule_list.append(cur_rule)

        if getattr(node, "children", None):
            for child in node.children:
                recur_add(child, rule_list)
    recur_add(tree, rule_list)
    return rule_list

def linearize_tree(tree):
    def recur_add(node):
        if getattr(node, "children", None) is None:
            return "{" + f"{node.value}" + "}"
        else:
            ret_str = f"[{node.data.value} "
            for child in node.children:
                ret_str += recur_add(child)
                ret_str += " "
            ret_str += "]"
            return ret_str
    return recur_add(tree)

def linearized_tree_to_program(linearized_tree, delimiter=""):
    tokens = re.findall(r'{(.*?)}', linearized_tree)
    return delimiter.join(tokens)

def normalize_program(program, parser):
    tree = parser.parse(program)
    linearized_tree = linearize_tree(tree)
    return linearized_tree_to_program(linearized_tree)

def rulelist2larkstr(rule_stat):
    lhs2rhs = collections.OrderedDict()
    for rule in rule_stat:
        lhs, rhs = rule.origin, rule.expansion
        if lhs not in lhs2rhs:
            lhs2rhs[lhs] = []
        lhs2rhs[lhs].append(rhs)
    
    grammar = ""
    for lhs in lhs2rhs:
        grammar += f"{lhs} :"
        for rhs in lhs2rhs[lhs]:
            rhs_str = " ".join(rhs)
            grammar += f" {rhs_str} |"
        grammar = grammar[:-2]
        grammar += "\n"
    
    return grammar.strip()

def rulelist2bnfstr(rule_list):
    """
    Convert list of rules to lark grammar string
    """
    larkstr = rulelist2larkstr(rule_list)
    bnf_str = lark2bnf(larkstr)
    return bnf_str

def extract_min_grammar_from_trees(trees, return_rules=False):
    """
    Extract minimal grammar to reconstruct the tree
    """
    rule_stat = collections.OrderedDict()
    for tree in trees:
        extract_rule_stat(tree, rule_stat)
    grammar = rulelist2larkstr(rule_stat)

    if return_rules:
        return grammar, list(rule_stat.keys())
    else:
        return grammar

def lark2bnf(grammar):
    """
    Make it easier for GPT to generate
    """
    #grammar = grammar.replace(" : ", " -> ")
    grammar = grammar.replace(" : ", " ::= ")
    return grammar

def bnf2lark(grammar):
    """
    Opposite of lark2bnf 
    """
    # grammar = grammar.replace(" -> ", " : ")
    grammar = grammar.replace(" ::= ", " : ")
    return grammar

def decorate_grammar(grammar):
    """
    Add auxiliary rules to the grammar
    """
    grammar += "\n%import common.DIGIT"
    grammar += "\n%import common.LCASE_LETTER"
    grammar += "\n%import common.UCASE_LETTER"
    grammar += "\n%import common.WS"
    grammar += "\n%ignore WS"
    return grammar

def collect_rules_from_examples(programs, parser):
    """
    Parse programs to extract rules and collect them. Mostly for debugging
    """
    rule_stat = collections.OrderedDict()
    for program in programs:
        tree = parser.parse(program)
        extract_rule_stat(tree, rule_stat)
    
    rulestr_set = set()
    for rule in rule_stat:
        rulestr = str(rule).strip()
        rulestr_set.add(rulestr)
    return rulestr_set

def collect_rules_from_larkfile(lark_file):
    """
    Parse bnf file (.lark) to extract rules
    """
    rule_stat = collections.OrderedDict() # used as ordered set
    aux_rules = []

    with open(lark_file, "r") as f:
        cur_nonterminal = None
        for line in f:
            line = line.strip()
            if line.startswith("%"):
                aux_rules.append(line)
            elif line == "" or line.startswith("//"):
                continue
            elif line.startswith("|"):
                rhs = line[1:].strip()
                for rhs_part in rhs.split("|"):
                    rhs_part = rhs_part.strip()
                    if rhs_part == "":
                        continue
                    assert cur_nonterminal is not None
                    rule = SimpleRule(cur_nonterminal, tuple(rhs_part.split()))
                    rule_stat[rule] = 1
            elif ":" in line and "\":" not in line: # for rules like :duration
                lhs, rhs = split_rule(line)
                cur_nonterminal = lhs
                for rhs_part in rhs.split("|"):
                    rhs_part = rhs_part.strip()
                    if rhs_part == "":
                        continue
                    rule = SimpleRule(cur_nonterminal, tuple(rhs_part.split()))
                    rule_stat[rule] = 1
            else:
                raise ValueError(f"Unknown line: {line}")
    rule_set = list(rule_stat.keys())
    return rule_set, aux_rules


def collect_rules_from_parser(parser, debug_rules=None):
    """
    Collect rules directly from parser. Note in some cases we 
    need to add " " to the terminal rules

    DEPRECATED unless updated
    TODO: currently I expand all terminals which is not good
    """
    def repattern2list(pattern):
        if pattern.type == "str":
            return [pattern.raw]
        else:
            re_stmt = pattern.value
            # unescape regex
            re_stmt = re_stmt.replace("\\", "")
            assert re_stmt[:3] == "(?:" and re_stmt[-1] == ")"
            elements = re_stmt[3:-1].split("|")
            return [f"\"{e}\"" for e in elements]
    
    rule_defs = parser.rules
    rule_set = set()
    for rule_def in rule_defs:
        origin = rule_def.origin.name.value

        catersian_product = []
        for nt_t in rule_def.expansion:
            if isinstance(nt_t, lark.grammar.Terminal):
                term_def = parser.get_terminal(nt_t.name)
                pattern = term_def.pattern
                candidates = repattern2list(pattern)
                catersian_product.append(candidates)
            elif isinstance(nt_t, lark.grammar.NonTerminal):
                catersian_product.append([nt_t.name])
        
        rhs_l = list(itertools.product(*catersian_product))
        for rhs in rhs_l:
            rule = SimpleRule(origin, list(rhs))
            rule_set.add(rule)
    
    # compress into string
    rulestr_set = set()
    for rule in rule_set:
        rulestr = str(rule).strip()
        rulestr_set.add(rulestr)
    
    if debug_rules:
        for rule in debug_rules:
            if rule not in rulestr_set: 
                import pdb; pdb.set_trace()
    return rulestr_set


def larkstr2rulelist(lark_str, rhs_sep=None):
    """
    Convert lark grammar string to list of rules.
    TODO: use load_grammar function from lark
    """
    for raw_rule in lark_str.split("\n"):
        lhs, rhs = split_rule(raw_rule)
        rhs_l = rhs.split("|")
        for rhs in rhs_l:
            rhs = rhs.strip()
            if rhs_sep is not None:
                rhs = rhs.split(rhs_sep)
                rule = SimpleRule(lhs, rhs)
            else:
                # treat rhs as a single token, which is enough 
                # for checking grammar validity bc. the the resulting string is the same
                rule = SimpleRule(lhs, (rhs,) )
            yield rule

def check_grammar_validity(valid_rules, pred_lark_str):
    """
    Check if the grammar (i.e., bnf_str produced by model) is valid
    """
    for rule in larkstr2rulelist(pred_lark_str):
        if rule.origin not in skipped_nonterminal_names and rule not in valid_rules:
            logger.debug(f"Found invalid rule {rule}")
            return False
    return True

def check_grammar_correctness(tgt_rules, pred_lark_str, debug=False):
    """
    Evaluate the correctness of the grammar
    """
    if pred_lark_str is None:
        return False
    tgt_ruleset = set(tgt_rules)
    pred_ruleset = set(larkstr2rulelist(pred_lark_str))

    if debug:
        logger.debug(f"Rules in pred but not in tgt: {pred_ruleset - tgt_ruleset}")
        logger.debug(f"Rules in tgt but not in pred: {tgt_ruleset - pred_ruleset}")

    return pred_ruleset == tgt_ruleset

def gen_min_lark(program, parser):
    """
    Obtain the minimal grammar from a program
    """
    parse_trees = []
    if "\n" in program:
        program = program.split("\n")
        for line in program:
            parse_tree = parser.parse(line)
            parse_trees.append(parse_tree)
    else:
        parse_tree = parser.parse(program)
        parse_trees.append(parse_tree)
    grammar = extract_min_grammar_from_trees(parse_trees)
    return grammar

def program2rules(program, parser):
    try:
        tree = parser.parse(program)
        rule_list = tree2rulelist(tree)
        return " ## ".join([rule.to_bnf() for rule in rule_list])
    except:
        # there are some bad cases, see run_parse_smc.py
        return program