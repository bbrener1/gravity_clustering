import numpy as np
import sys
import os
from pathlib import Path
import io
import argparse
from time import sleep
import subprocess as sp

import numpy as np
# import matplotlib.pyplot as plt

def main():
    print("Running main")
    counts = np.loadtxt(sys.argv[1])
    fit_predict(counts,scaling=.1,sample_sub=10)

def fit_predict(targets,command,feature_sub=None,distance=None,sample_sub=None,scaling=None,merge_distance=None,refining=False,error_dump=None,convergence_factor=None,smoothing=None,locality=None):

    # np.array(targets)

    targets = "\n".join(["\t".join([str(y) for y in x]) for x in targets]) + "\n"

    # print(targets)

    # table_pipe = io.StringIO(targets)
    #
    # table_pipe.seek(0)

    path_to_rust = (Path(__file__).parent / "target/release/gravity_clustering").resolve()

    print("Running " + str(path_to_rust))

    arg_list = [str(path_to_rust),command]
    arg_list.extend(["-stdin"])
    arg_list.extend(["-stdout"])
    if sample_sub is not None:
        arg_list.extend(["-ss",str(sample_sub)])
    if feature_sub is not None:
        arg_list.extend(["-fs",str(feature_sub)])
    if scaling is not None:
        arg_list.extend(["-sf",str(scaling)])
    if merge_distance is not None:
        arg_list.extend(["-m",str(merge_distance)])
    if error_dump is not None:
        arg_list.extend(["-error",str(error_dump)])
    if convergence_factor is not None:
        arg_list.extend(["-convergence",str(convergence_factor)])
    if locality is not None:
        arg_list.extend(["-l",str(locality)])
    if smoothing is not None:
        arg_list.extend(["-smoothing",str(smoothing)])
    if distance is not None:
        arg_list.extend(["-d",str(distance)])
    if refining:
        arg_list.extend(["-refining"])

    print("Command: " + " ".join(arg_list))

    cp = sp.run(arg_list,input=targets,stdout=sp.PIPE,stderr=sp.PIPE,universal_newlines=True)

    while cp.returncode is None:
        cp.poll()
        sleep(1)
        print(cp.stderr.read())

    print(cp.stderr)

    # print(cp.stdout)

    return(list(map(lambda x: int(x),cp.stdout.split())))


if __name__ == "__main__":
    main()
