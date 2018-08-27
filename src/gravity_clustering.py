import numpy as np
import sys
import os
import io
import argparse

import subprocess as sp

def fit_predict(targets,feature_sub=None,sample_sub=None,scaling=None,):

    table_pipe = io.StringIO()
    np.savetxt(table_pipe,targets)

    arg_string = ["./target/release/gravity_clustering"]
    arg_string.extend("-")

    sp.run()
