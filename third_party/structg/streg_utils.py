def tokenize_specification(x):
    y = []
    while len(x) > 0:
        head = x[0]
        if head in ["(", ")", ","]:
            y.append(head)
            x = x[1:]
        elif head == "<":
            end = x.index(">") + 1
            y.append(x[:end])
            x = x[end:]
        else:
            leftover = [(i in ["(", ")", "<", ">", ","]) for i in x]
            end = leftover.index(True)
            y.append(x[:end])
            x = x[end:]
    return y

def _consume_an_ast_node(tokens, cursor):
    pass

# _NODE_CLASS_TO_RULE = {
#     "not": "Not",
#     "notcc": "NotCC",
#     "star": "Star",
#     "optional": "Optional",
#     "startwith": "StartWith",
#     "endwith": "EndWith",
#     "contain": "Contain",
#     "concat": "Concat",
#     "and": "And",
#     "or": "Or",
#     "repeat": "Repeat",
#     "repeatatleast": "RepeatAtleast",
#     "repeatrange": "RepeatRange",
#     "const": "String"
# }

def parse_spec_toks_to_ast(tokens):
    ast, final_cursor = _parse_spec_toks_to_ast(tokens, 0)
    assert final_cursor == len(tokens)
    return ast

def _parse_spec_toks_to_ast(tokens, cursor):
    cur_tok = tokens[cursor]
    # unary operator
    if cur_tok in ['not', 'notcc', 'star', 'optional', 'startwith', 'endwith', 'contain']:
        assert tokens[cursor + 1] == '('
        child, cursor = _parse_spec_toks_to_ast(tokens, cursor + 2)
        assert tokens[cursor] == ')'
        cursor += 1
        node = StRegNode(cur_tok, [child])
    elif cur_tok in ['and', 'or', 'concat']:
        assert tokens[cursor + 1] == '('
        left_child, cursor = _parse_spec_toks_to_ast(tokens, cursor + 2)
        assert tokens[cursor] == ','
        right_child, cursor = _parse_spec_toks_to_ast(tokens, cursor + 1)
        assert tokens[cursor] == ')'
        cursor += 1
        node = StRegNode(cur_tok, [left_child, right_child])
    elif cur_tok in ['repeat', 'repeatatleast']:
        assert tokens[cursor + 1] == '('
        child, cursor = _parse_spec_toks_to_ast(tokens, cursor + 2)
        assert tokens[cursor] == ','
        assert tokens[cursor + 1].isdigit()
        int_val = int(tokens[cursor + 1])
        assert tokens[cursor + 2] == ')'
        cursor = cursor + 3
        node = StRegNode(cur_tok, [child], [int_val])
    elif cur_tok in ['repeatrange']:
        assert tokens[cursor + 1] == '('
        child, cursor = _parse_spec_toks_to_ast(tokens, cursor + 2)
        assert tokens[cursor] == ','
        assert tokens[cursor + 1].isdigit()
        int_val1 = int(tokens[cursor + 1])
        assert tokens[cursor + 2] == ','
        assert tokens[cursor + 3].isdigit()
        int_val2 = int(tokens[cursor + 3])
        cursor = cursor + 4
        node = StRegNode(cur_tok, [child], [int_val1, int_val2])
    elif cur_tok.startswith('<') and cur_tok.endswith('>'):
        cursor += 1
        node = StRegNode(cur_tok)
    # not really a valid ast, need to be replaced with real const
    elif cur_tok.startswith('const'):
        cursor += 1
        node = StRegNode(cur_tok)
    else:
        raise RuntimeError('Not parsable', cur_tok)
    return node, cursor

# pasre a specification to AST
def parse_spec_to_ast(x):
    toks = tokenize_specification(x)
    ast = parse_spec_toks_to_ast(toks)
    assert x == ast.logical_form()
    return ast


#  ASTNoode
# node_class: the name of nonterminal or terminal
# children: list of children nodes
# params: intergers for repeat/repeatatleast/repeatrange
class StRegNode:
    def __init__(self, node_class, children=[], params=[]):
        self.node_class = node_class
        self.children = children
        self.params = params
    
    def logical_form(self):
        if len(self.children) + len(self.params) > 0:
            return self.node_class + "(" + ",".join([x.logical_form() for x in self.children] + [str(x) for x in self.params]) + ")"
        else:
            return self.node_class

    def debug_form(self):
        if len(self.children) + len(self.params) > 0:
            return str(self.node_class) + "(" + ",".join([x.debug_form() if x is not None else str(x) for x in self.children] + [str(x) for x in self.params]) + ")"
        else:
            return str(self.node_class)
    
    def short_debug_form(self):
        x = self.debug_form()
        tunct_pair = [('None', '?'), ('concat', 'cat'), ('repeatatleast', 'rp+'), ('repeatrange', 'rprng'), ('repeat', 'rp'), ('optional', 'optn')]
        # x = x.replace('concat', 'cat')
        for a, b in tunct_pair:
            x = x.replace(a, b)
        return x

    def tokenized_logical_form(self):
        if len(self.children) + len(self.params) > 0:
            toks = [self.node_class] + ["("]
            toks.extend(self.children[0].tokenized_logical_form())
            for c in self.children[1:]:
                toks.append(",")
                toks.extend(c.tokenized_logical_form())
            for p in [str(x) for x in self.params]:
                toks.append(",")
                toks.append(p)
            toks.append(")")
            return toks
        else:
            return [self.node_class]
    
    # some operators can't be converted: not, and
    def standard_regex(self):
        if self.node_class == '<let>':
            return '[A-Za-z]'
        elif self.node_class == '<num>':
            return '[0-9]'
        elif self.node_class == 'concat':
            return '(%s)(%s)' % (self.children[0].standard_regex(), self.children[1].standard_regex())
        elif self.node_class == 'contain':
            return '.*(%s).*' % (self.children[0].standard_regex())
        elif self.node_class == 'repeatatleast':
            return '(%s){%d,}' % (self.children[0].standard_regex(), self.params[0])
        else:
            # add code for parsing other terminals and operators
            raise NotImplementedError('Please fill in')
