from third_party.structg.eval import check_equiv, check_io_consistency
from third_party.structg.streg_utils import parse_spec_to_ast

# check equivalance
print('EXPECTED TRUE', check_equiv('or(<low>,<cap>)', 'or(<cap>,<low>)'))
print('EXPECTED TRUE',check_equiv('or(<low>,<cap>)', '<let>'))
print('EXPECTED FALSE',check_equiv('concat(<low>,<cap>)', 'concat(<cap>,<low>)'))

# check example consistency

spec = 'and(repeatatleast(or(<low>,or(<cap>,<^>)),1),and(not(startwith(<low>)),and(not(startwith(<^>)),not(contain(concat(notcc(<low>),<^>))))))'
good_examples = [('ItrdY', '+'), ('JIQD', '+'), ('GAFXvIc^j^l^o^op', '+'), ('WZpg^y^eMrXSfXTqHw^', '+'), ('Y', '+'), ('Jw', '+'), ('cvZpBMcQKAqAXj', '-'), ('X^^mwwSbU^Wk^', '-'), ('ZHQgmLzM^', '-'), ('.-;-g', '-'), (':;A:', '-'), ('Ew^^B^Kcc^zR', '-')]
bad_examples1 = [('123ItrdY', '+'), ('ItrdY', '+'), ('JIQD', '+'), ('GAFXvIc^j^l^o^op', '+'), ('WZpg^y^eMrXSfXTqHw^', '+'), ('Y', '+'), ('Jw', '+'), ('cvZpBMcQKAqAXj', '-'), ('X^^mwwSbU^Wk^', '-'), ('ZHQgmLzM^', '-'), ('.-;-g', '-'), (':;A:', '-'), ('Ew^^B^Kcc^zR', '-')]
bad_examples2 = [('ItrdY', '-'), ('JIQD', '+'), ('GAFXvIc^j^l^o^op', '+'), ('WZpg^y^eMrXSfXTqHw^', '+'), ('Y', '+'), ('Jw', '+'), ('cvZpBMcQKAqAXj', '-'), ('X^^mwwSbU^Wk^', '-'), ('ZHQgmLzM^', '-'), ('.-;-g', '-'), (':;A:', '-'), ('Ew^^B^Kcc^zR', '-')]

print('EXPECTED TRUE',check_io_consistency(spec, good_examples))
print('EXPECTED FALSE',check_io_consistency(spec, bad_examples1))
print('EXPECTED FALSE',check_io_consistency(spec, bad_examples2))


# skeleton for converting to standard regex
print('concat(<let>,<num>)', parse_spec_to_ast('concat(<let>,<num>)').standard_regex())
print('contain(<let>)', parse_spec_to_ast('contain(<let>)').standard_regex())
print('repeatatleast(concat(<let>,<num>),3)', parse_spec_to_ast('repeatatleast(concat(<let>,<num>),3)').standard_regex())


ast = parse_spec_to_ast(spec)
print(ast.logical_form())
std_regex = ast.standard_regex()
