import re
import stanza
import collections

from minEarley.tree import Tree
from stanza.models.constituency.parse_tree import Tree as StanzaTree
from dataclasses import dataclass
from neural_lark.train_utils import logger
from neural_lark.lark_utils import rulelist2larkstr, rulelist2bnfstr, SimpleRule, larkstr2rulelist

"""
Apart from larkstr and bnfstr, we have normalstr which are rules using the original PTB symbols. Internally, we need to convert them to larkstr (not bnfstr) for parsing.
"""


def load_symbols(filename):
    sym2desc = {}
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            sym, desc = line.split(" ")
            sym2desc[sym] = desc
    return sym2desc

PT2DECS = load_symbols("grammars/ptb/pos.txt")
NT2DECS = load_symbols("grammars/ptb/nt.txt")


def treenode2rule(treenode):
    def normalize(symbol):
        if symbol in [",", ".", ":", "``", "''", "-LRB-", "-RRB-"]:
            symbol = "PUNCT"
        elif symbol[-1] == "$":
            symbol = symbol[:-1] + "2"
        return symbol

    if treenode is None:
        return None
    assert isinstance(treenode, StanzaTree)

    origin = normalize(treenode.label)
    expansion = []
    for child in treenode.children:
        if child.is_leaf():
            expansion.append(f"\"{child.label}\"")
        elif child.is_preterminal(): 
            label = normalize(child.label)
            expansion.append(label)
        else:
            label = normalize(child.label)
            expansion.append(label)
    return SimpleRule(origin, tuple(expansion))

def normalize_larkstr(normal_str):
    """
    Normalize the larkstr to be compatible with Lark
        - use lower case for non-terminals
        - use "start" for ROOT
    """
    def normalize(symbol):
        if symbol in NT2DECS:
            symbol = symbol.lower()
        elif symbol == "ROOT":
            symbol = "start"
        return symbol
    
    rules = list(larkstr2rulelist(normal_str, rhs_sep=" "))
    for rule in rules:
        rule.origin = normalize(rule.origin)
        rule.expansion = tuple([normalize(symbol) for symbol in rule.expansion])
    return rulelist2larkstr(rules)
    
    
def extract_rule_stat(tree, rule_stat):
    """
    Count the occurrence of each rule
    """
    cur_rule = treenode2rule(tree)
    if tree.is_leaf():
        return
    if cur_rule not in rule_stat:
        rule_stat[cur_rule] = 1
    else:
        rule_stat[cur_rule] += 1

    if getattr(tree, "children", None):
        for child in tree.children:
            extract_rule_stat(child, rule_stat)

def truncate_tree(tree, num_limit=5, length_limit=10):
    # breadth-first traversal
    queue = collections.deque([tree])
    while queue:
        cur_node = queue.popleft()
        if getattr(cur_node, "children", None):
            for child in cur_node.children:
                queue.append(child)

        # if cur_node.label in ["ROOT", "S", "NP", "PP", "VP"]:
        if not cur_node.is_leaf() and (len(cur_node.leaf_labels()) <= length_limit or num_limit < 0):
            new_child = " ".join(cur_node.leaf_labels())
            new_child = re.sub(r'\s([,.:;](?:\s|$))', r'\1', new_child)
            cur_node.children = [StanzaTree(new_child, [])]
            new_child = " ".join(cur_node.leaf_labels())
        else:
            num_limit -= 1

            # if cur_node.label != "ROOT":  
            #     new_child = " ".join(cur_node.leaf_labels())
            #     new_child = re.sub(r'\s([,.:;](?:\s|$))', r'\1', new_child)
            #     cur_node.label = new_child
            #     cur_node.children = []


def extract_min_grammar_from_trees(trees, return_rules=False):
    """
    Extract minimal grammar to reconstruct the tree
    """
    rule_stat = collections.OrderedDict()
    for tree in trees:
        truncate_tree(tree)
        extract_rule_stat(tree, rule_stat)
    
    def update_root_rule(rules):
        to_delete = []
        for rule in rules:
            if rule.origin == "ROOT":
                to_delete.append(rule)
        for rule in to_delete:
            del rule_stat[rule]

        root_rule = SimpleRule("ROOT", tuple(tree.children[0].label for tree in trees))
        rule_stat.update({root_rule: 1})
        rule_stat.move_to_end(root_rule, last=False)

    if len(trees) > 1:
        all_rules = rule_stat.keys()
        update_root_rule(all_rules)

    grammar = rulelist2larkstr(rule_stat)
    if return_rules:
        return grammar, list(rule_stat.keys())
    else:
        return grammar