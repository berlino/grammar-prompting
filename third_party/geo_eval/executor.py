"""An executor for GeoQuery FunQL programs."""

from overrides import overrides
from pyswip import Prolog
import re

class ProgramExecutorGeo:

    def __init__(self):
        self._prolog = Prolog()
        self._prolog.consult('third_party/geo_eval/geobase.pl')
        self._prolog.consult('third_party/geo_eval/geoquery.pl')
        self._prolog.consult('third_party/geo_eval/eval.pl')

    def execute(self, program: str, kb_str: str = None) -> str:
        # make sure entities with multiple words are parsed correctly
        program = re.sub("' (\w+) (\w+) '", "'"+r"\1#\2"+"'", program)
        program = re.sub("' (\w+) (\w+) (\w+) '", "'" + r"\1#\2#\3" + "'", program)
        program = program.replace(' ','').replace('#',' ')

        try:
            answers = list(self._prolog.query("eval(" + '{}, X'.format(program) + ").", maxresult=1))
        except Exception as e:
            return 'error_parse: {}'.format(e)
        return str([str(answer) for answer in answers[0]['X']])


if __name__ == "__main__":
    pred_program = "answer ( population_1 ( city ( loc_2 ( stateid ( 'minnesota' ) ) ) ) )"
    executor = ProgramExecutorGeo()
    denotation = executor.execute(pred_program)
    print(denotation)
