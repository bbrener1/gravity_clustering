import numpy as np
import sys
import os
from pathlib import Path
import io
import argparse

import subprocess as sp

import numpy as np
# import matplotlib.pyplot as plt

def main():
    print("Running main")
    counts = np.loadtxt(sys.argv[1])
    fit_predict(counts,scaling=.1,sample_sub=10)

def fit_predict(targets,feature_sub=None,sample_sub=None,scaling=None,):

    # np.array(targets)

    targets = "\n".join(["\t".join([str(y) for y in x]) for x in targets]) + "\n"

    print(targets)

    # table_pipe = io.StringIO(targets)
    #
    # table_pipe.seek(0)

    path_to_rust = (Path(__file__).parent / "target/release/gravity_clustering").resolve()

    print("Running " + str(path_to_rust))

    arg_list = [str(path_to_rust),"fitpredict"]
    arg_list.extend(["-stdin"])
    arg_list.extend(["-stdout"])
    if sample_sub is not None:
        arg_list.extend(["-ss",str(sample_sub)])
    if scaling is not None:
        arg_list.extend(["-sf",str(scaling)])


    cp = sp.run(arg_list,input=targets,stdout=sp.PIPE,stderr=sp.PIPE,universal_newlines=True)

    # print(cp.stderr)

    # print(cp.stdout)

    return(list(map(lambda x: int(x),cp.stdout.split())))


if __name__ == "__main__":
    main()
