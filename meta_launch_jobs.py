import numpy as np
import pdb
import os
import time
import sys


PRIORS = ['vampprior_short']
REPLAYS = ['increase']
ADD_CAP = [1]
#CLASSIFIER = [1]
REBALANCES = [0, 1]
VAMP_MIX = [1]


# dopnt iterate through them now
add_cap = ADD_CAP[0]
vamp_mix = VAMP_MIX[0]
dynamic_binarization = [1]

for prior in PRIORS:
  for replay in REPLAYS:
    for rebalance in REBALANCES:  
        for db in dynamic_binarization:
            command="main_mnist.py \
                --prior %(prior)s \
                --replay_size %(replay)s \
                --add_cap %(add_cap)s \
                --use_vampmixingw %(vamp_mix)s \
                --use_mixingw_correction %(rebalance)s \
                --dynamic_binarization %(db)s" % locals()
            
            print(command)
        
            command = "{} cc_launch_cl.sh {}".format(sys.argv[1], command) 

            os.system(command)
            time.sleep(2)












