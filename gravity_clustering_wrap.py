import numpy as np
import sys
import os
import tempfile as tmp
from pathlib import Path
import io
import argparse
from time import sleep
import subprocess as sp
from threading import Thread
from queue import Queue, Empty
# import asyncio as aio
import numpy as np
# import matplotlib.pyplot as plt

def main():
    print("Running main")
    counts = np.loadtxt(sys.argv[1])
    fit_predict(counts,scaling=.1,sample_sub=10)

def fit_predict(targets,command,feature_sub=None,distance=None,verbose=False,sample_sub=None,scaling=None,refining=False,fuzz=None,step_fraction=None,steps=None,borrow=None,error_dump=None,convergence_factor=None,smoothing=None,locality=None):

    # np.array(targets)
    targets = targets.astype(dtype=float)

    targets = "\n".join(["\t".join([str(y) for y in x]) for x in targets]) + "\n"


    input_temp = tmp.NamedTemporaryFile()
    progress_temp = tmp.NamedTemporaryFile()
    # final_pos_temp = tmp.NamedTemporaryFile()

    input_writer = open(input_temp.name,mode='w')
    input_writer.write(targets)
    input_writer.close()

    # print(targets)

    # table_pipe = io.StringIO(targets)
    #
    # table_pipe.seek(0)

    path_to_rust = (Path(__file__).parent / "target/release/gravity_clustering").resolve()

    print("Running " + str(path_to_rust))

    arg_list = [str(path_to_rust),command]
    arg_list.extend(["-c",input_temp.name])
    # arg_list.extend(["-stdin"])
    # arg_list.extend(["-stdout"])
    if verbose:
        arg_list.extend(["-verbose"])
    if sample_sub is not None:
        arg_list.extend(["-ss",str(sample_sub)])
    if feature_sub is not None:
        arg_list.extend(["-fs",str(feature_sub)])
    if scaling is not None:
        arg_list.extend(["-sf",str(scaling)])
    if fuzz is not None:
        arg_list.extend(["-fuzz",str(fuzz)])
    if error_dump is not None:
        arg_list.extend(["-error",str(error_dump)])
    if convergence_factor is not None:
        arg_list.extend(["-convergence",str(convergence_factor)])
    if step_fraction is not None:
        arg_list.extend(["-step_fraction",str(step_fraction)])
    if steps is not None:
        arg_list.extend(["-steps",str(steps)])
    if locality is not None:
        arg_list.extend(["-l",str(locality)])
    if smoothing is not None:
        arg_list.extend(["-smoothing",str(smoothing)])
    if distance is not None:
        arg_list.extend(["-d",str(distance)])
    if borrow is not None:
        arg_list.extend(["-borrow",str(borrow)])
    if refining:
        arg_list.extend(["-refining"])
    arg_list.extend(["2>"+progress_temp.name])

    print("Command: " + " ".join(arg_list))

    # print("Peek at input:")
    # print(open(input_temp.name,mode='r').read())

    # cp = sp.Popen(arg_list,stdin=sp.PIPE,stdout=sp.PIPE,stderr=sp.PIPE)
    cp = sp.Popen(" ".join(arg_list),stdout=sp.PIPE,universal_newlines=True,shell=True)

    # cp.stdin.write(targets.encode())

    # cp = sp.run(arg_list,input=targets,stdout=sp.PIPE,stderr=sp.PIPE,universal_newlines=True)
    # cp = sp.run(" ".join(arg_list),input=targets,stdout=sp.PIPE,stderr=sp.PIPE,universal_newlines=True,shell=True)

    # while True:
    #     # sleep(0.1)
    #     rc = cp.poll()
    #     if rc is not None:
    #         for line in cp.stderr.readlines():
    #             sleep(.001)
    #             print(line)
    #         break
    #     output = cp.stderr.readline()
    #     # print("Read line")
    #     print(output.strip())
    #

    progress_counter = 0

    while cp.poll() is None:
        sleep(.01)
        line = progress_temp.readline()
        if verbose and line != b"":
            print(line,flush=True)
            # print(line.count(b's:'),flush=True)
        #     progress_counter += line.count(b's:')
        #     print(f"Points descended:{progress_counter}",flush=True)
        # else:
        if not verbose:
            progress_counter += line.count(b's:')
            if b"Clusters" not in line:
                print(f"Points descended:{progress_counter}",end="\r",flush=True)
            else:
                print(str(line.strip), end='\r')
        # if line != b"":
        #     print(line)
        # else:
        #     print(cp.returncode)

    print("Broke loop")

    for line in progress_temp.readlines():
        print(line)

    # print(cp.stdout.read())

    return(list(map(lambda x: int(x),cp.stdout.read().split())))
