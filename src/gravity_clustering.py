import numpy as np
import sys
import os
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

    print(targets.shape)
    print(targets[:2])

    table_pipe = io.StringIO()

    for sample in targets:
        table_pipe.write('\t'.join([str(x) for x in sample.tolist()]) + '\n')

    table_pipe.seek(0)

    print(table_pipe.getvalue())

    arg_list = ["./target/release/gravity_clustering","fitpredict"]
    arg_list.extend(["-stdin"])
    arg_list.extend(["-stdout"])
    arg_list.extend(["-ss",str(sample_sub)])
    arg_list.extend(["-sf",str(scaling)])


    cp = sp.run(arg_list,input=table_pipe.read(),stdout=sp.PIPE,universal_newlines=True)



    return(list(map(lambda x: int(x),cp.stdout.split())))


if __name__ == "__main__":
    main()
