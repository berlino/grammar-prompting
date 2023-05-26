import os
from os.path import join
import subprocess

from os.path import abspath, dirname
path = dirname(__file__)

def check_equiv(spec0, spec1):
    if spec0 == spec1:
        # print("exact", spec0, spec1)
        return True
    # try:
    out = subprocess.check_output(
        ['java', '-cp', f'{path}/external/datagen.jar:{path}/external/lib/*', '-ea', 'datagen.Main', 'equiv',
            spec0, spec1], stderr=subprocess.DEVNULL)
    out = out.decode("utf-8")
    out = out.rstrip()
    # if out == "true":
    #     print("true", spec0, spec1)

    return out == "true"

# examples: (xxx,'+), (xxx, '-')
def check_io_consistency(spec, examples):
    # pred_line = " ".join(preds)
    pred_line = "{} {}".format(spec, spec)
    exs_line = " ".join(["{},{}".format(x[1], x[0]) for x in examples])

    try:
        out = subprocess.check_output(
            ['java', '-cp', f'{path}/external/datagen.jar:{path}/external/lib/*', '-ea', 'datagen.Main', 'preverify',
                pred_line, exs_line], stderr=subprocess.DEVNULL, timeout=5)
    except subprocess.TimeoutExpired as e:
        return False

    # stderr=subprocess.DEVNULL    
    out = out.decode("utf-8")
    out = out.rstrip()
    # print(streg_ast.debug_form())
    return out == "true"
