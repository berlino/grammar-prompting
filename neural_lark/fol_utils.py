from neural_lark.logic import *
from minEarley.tree import Tree
from lark.lexer import Token

from neural_lark.train_utils import logger

def v_args(inline=False):
    """
    If inline is False, pack all args into a list (i.e., args).
    Else, pass as is.
    """
    def decorator(func):
        def wrapper(self, *args):
            if inline:
                return func(self, *args)
            else:
                return func(self, args)

        return wrapper

    return decorator

class FOLTransformer:
    """
    Mindful of whether the input and output should be a list of unpacked list. 
    """

    def _call_user_func(self, tree, args):
        """
        Args:
            args: new children
        """
        try:
            f = getattr(self, tree.data.value)
            return f(args)
        except AttributeError:
            return tree
    
    def _handle_token(self, token):
        return token.value
    
    def _transform_tree(self, tree):
        args = []
        for child in tree.children:
            if isinstance(child, Tree):
                arg = self._transform_tree(child)
            elif isinstance(child, Token):
                arg = self._handle_token(child)
            else:
                arg = child
            args.append(arg)
        
        if len(args) == 1:
            args = args[0]

        return self._call_user_func(tree, args)
    
    def transform(self, tree):
        return self._transform_tree(tree)

    @v_args(inline=True)
    def formula(self, arg):
        return arg

    @v_args(inline=True)
    def atomic_formula(self, args):
        if not isinstance(args, list):
            return Atom(args)
        else:
            assert args[1] == "(" and args[-1] == ")"
            term_list = args[2]
            if isinstance(term_list, list):
                return Atom(args[0], *term_list)
            else:
                return Atom(args[0], term_list)

    @v_args(inline=True)
    def complex_formula(self, args):
        if args[0] == "(" and args[-1] == ")":
            return args[1]
        elif args[0] == "¬":
            return Not(args[1])
        elif args[1] in ["∧", ","]:
            return And(args[0], args[2])
        elif args[1] == "∨":
            return Or(args[0], args[2])
        elif args[1] == "→":
            return Implies(args[0], args[2])
        elif args[1] in ["↔", "⟷"]:
            return Equiv(args[0], args[2])
        elif args[1] == "⊕":
            return Xor(args[0], args[2])
        elif args[0] == "∀":
            return Forall(args[1], args[2])
        elif args[0] == "∃":
            return Exists(args[1], args[2]) 
        else:
            raise ValueError(f"Unknown complex formula: {args}")

    @v_args(inline=True)
    def operator(self, arg):
        return arg 

    @v_args(inline=True)
    def quantifier(self, arg):
        return arg
    
    @v_args(inline=True)
    def predicate(self, arg):
        return arg
    
    @v_args(inline=True)
    def constant(self, arg):
        if not arg[0].islower():
            logger.warning(f"Constant {arg} is not lower case.")
        return Constant(arg)
    
    @v_args(inline=True)
    def variable(self, arg):
        return Variable(f"${arg}")
    
    @v_args(inline=True)
    def term_list(self, args):
        # term_list ::= term
        if not isinstance(args, list):
            return args
        # term_list ::= term "," term_list
        else:
            res = [arg for arg in args if arg != ","]
            return res

    @v_args(inline=True)
    def term(self, arg):
        return arg

def cleanup_fol(nl):
    formula_strs = nl.split('\n')

    # sometime gpt3.5 will generate comments
    cleaned_formula_strs = []
    for idx in range(len(formula_strs)):
        formula_str = formula_strs[idx]
        formula_str = formula_str.strip()

        if formula_str in ["```", "*/"]:
            continue
        formula_str = formula_str.split("//")[0]
        formula_str = formula_str.split("#")[0]
        cleaned_formula_strs.append(formula_str)
    formula_strs = cleaned_formula_strs
    return "\n".join(formula_strs)

def parse_fol(nl, parser):
    nl = cleanup_fol(nl)
    formula_strs = nl.split('\n')

    formulas = []
    transformer = FOLTransformer()
    for formula_str in formula_strs:
        parse = parser.parse(formula_str)
        formula = transformer.transform(parse)
        formulas.append(formula)
    premises, conclusion = formulas[:-1], formulas[-1]
    return premises, conclusion

def execute_fol(premises, conclusion):
    kb = createResolutionKB()
    for premise in premises:
        kb.tell(premise)
    res = kb.ask(conclusion)
    return res

def check_kb(premises):
    kb = createResolutionKB()
    for premise in premises:
        kb.tell(premise)
    return kb

