#!/usr/bin/env python

import subprocess
import os
import glob


exps = ["/tmp/exp1", "/tmp/exp2", "/tmp/exp3"]

for exp in exps:
    yaml_fn = glob.glob(os.path.join(exp, '*.yaml'))
    yaml_fn.sort()

    cmd = "cd {} ; ".format(os.path.abspath('..'))

    if len(yaml_fn) > 0:
        cmd += "python tools/train_net.py --cfg {} OUTPUT_DIR {}  | tee {}/log.txt".format(
            yaml_fn[0], exp, exp)
        print(cmd)
        subprocess.call(cmd, shell=True)
    else:
        print("In Dir:{}, yaml not found.".format(exp))

print("all done.")
