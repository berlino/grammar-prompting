import json
import os
import tempfile
import subprocess
from nltk import tree

_rule_path = "grammars/overnight/terminals.json"
with open(_rule_path, "r") as f:
    overnight_rules = json.load(f)

def norm_value(vp: str):
    # use '#' to pack and unpack terminals
    # return "#".join(vp.split())
    return vp 

def denorm_value(vp: str):
    # return " ".join(vp.split("#"))
    return vp

def to_lisp_like_string(node):
    if isinstance(node, tree.Tree):
        return f"({node.label()} {' '.join([to_lisp_like_string(child) for child in node])})"
    else:
        return node

def remove_lf_space(raw_lf: str):
    """
    See run_sempar_icl.py for usage
    """
    try:
        lf_tree = tree.Tree.fromstring(raw_lf)
        return to_lisp_like_string(lf_tree)
    except:
        return raw_lf


def normalize_lf(raw_lf: str):
    """
    1. predicates and entities are realized by string in simpleword.java, we use property, value, string to differentiate them
    2. remove the terminal functions like string, number
    """
    lf_tree = tree.Tree.fromstring(raw_lf)

    def normalize(node):
        if isinstance(node, tree.Tree) and node.label() == "call":
            # map the SW functions to our defined ones
            if node[0].startswith("SW."):
                node.set_label(node[0][3:])
            elif node[0].startswith("."):  # .size
                node.set_label("_" + node[0][1:])  # _size
            else:
                raise NotImplementedError
            node.remove(node[0])  # remove call

            for index, child in enumerate(node):
                if (
                    isinstance(child, tree.Tree)
                    and child.label() in overnight_rules["terminal_types"]
                ):
                    # particularly handle the terminals of properties
                    raw_child_str = " ".join(child)
                    norm_child_str = norm_value(raw_child_str)
                    node[index] = norm_child_str
                elif isinstance(child, tree.Tree) and child.label() == "":
                    assert len(child) == 2

                    # replace the lambda with its first child
                    child[0] = child[0][1]
                    assert child[0][1].leaves() == ["s"]
                    # replace the variable
                    child[0][1] = child[1]
                    # replace the lambda node with new grounded node
                    node[index] = child[0]
                    normalize(node[index])
                else:
                    normalize(child)

    normalize(lf_tree)
    normalized_lf = to_lisp_like_string(lf_tree)

    normalized_lf = normalized_lf.replace("! type", "!type")
    normalized_lf = normalized_lf.replace("! =", "!=")
    return normalized_lf

def denormalize_lf(raw_lf: str):
    """
    Opposite of normalize_lf
    """
    function_names = ["listValue", "filter", "ensureNumericProperty", "ensureNumericEntity", "superlative", "countSuperlative", "countComparative", "_size", "aggregate", "getProperty", "singleton", "concat"]

    string_symbols = ["shape", "color", "length", "is_special", "width", "height", "left", "right", "above", "below", "=", ">", "<", ">=", "<=", "!=", "sum", "max", "min", "avg", "!type"]

    number_symbols = ["3 en.inch", "6 en.inch", "2"]
    padded_number_symbols = [symbol.replace(" ", "#") for symbol in number_symbols]

    for symbol in number_symbols:
        if symbol in raw_lf:
            pad_symbol = symbol.replace(" ", "#")
            raw_lf = raw_lf.replace(symbol, pad_symbol)

    lf_tree = tree.Tree.fromstring(raw_lf)

    def denormalize(node):
        if isinstance(node, tree.Tree) and node.label() in function_names: 
            # change the way of calling functions
            if node.label().startswith("_"):  # _size
                real_label = "." + node.label()[1:]
            else:
                real_label = "SW." + node.label()
            node.set_label("call")
            node.insert(0, real_label)

            for index, child in enumerate(node):
                if index == 0:
                    continue  # func name
                elif not isinstance(child, tree.Tree):
                    # denormalize the terminals
                    if child in string_symbols:
                        node[index] = f"(string {child})"
                    elif child in padded_number_symbols:
                        child = child.replace("#", " ")
                        node[index] = f"(number {child})"
                    else:
                        pass
                else:
                    denormalize(child)

    def to_lisp_like_string(node):
        if isinstance(node, tree.Tree):
            return f"( {node.label()} {' '.join([to_lisp_like_string(child) for child in node])} )"
        else:
            return node

    denormalize(lf_tree)
    denormalized_lf = to_lisp_like_string(lf_tree)
    return denormalized_lf

def execute(lfs, domain, eval_path="third_party/overnight"):
    def post_process(lf):
        if lf is None:
            lf = "None"
        replacements = [("SW", "edu.stanford.nlp.sempre.overnight.SimpleWorld")]
        for a, b in replacements:
            lf = lf.replace(a, b)
        return lf

    def is_error(d):
        return "FAILED" in d or "Exception" in d

    cur_dir = os.getcwd()
    os.chdir(eval_path)
    eval_script = "./evaluator/overnight"

    tf = tempfile.NamedTemporaryFile(suffix=".examples")
    for lf in lfs:
        p_lf = post_process(lf)
        tf.write(str.encode(p_lf + "\n"))
        tf.flush()
    FNULL = open(os.devnull, "w")
    msg = subprocess.check_output([eval_script, domain, tf.name], stderr=FNULL)
    tf.close()
    msg = msg.decode("utf-8")

    denotations = [
        line.split("\t")[1]
        for line in msg.split("\n")
        if line.startswith("targetValue\t")
    ]
    denotations = [None if is_error(d) else d for d in denotations]
    os.chdir(cur_dir)
    return denotations

if __name__ == "__main__":
    raw_lf = "( call SW.listValue ( call SW.filter ( call SW.getProperty ( call SW.singleton en.block ) ( string !type ) ) ( string right ) ( string = ) ( call SW.filter ( call SW.getProperty ( call SW.singleton en.block ) ( string !type ) ) ( string height ) ( string != ) ( number 3 en.inch ) ) ) )"
    print(raw_lf)

    norm_lf = normalize_lf(raw_lf)
    # print(norm_lf)

    denorm_lf = denormalize_lf(norm_lf)
    print(denorm_lf)

    denotations = execute([denorm_lf], "blocks")
    print(denotations)