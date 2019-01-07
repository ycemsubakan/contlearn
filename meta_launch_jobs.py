import numpy as np
import pdb
import os
import time
import sys


PRIORS = ['vampprior_short', 'standard']
REPLAYS = ['increase', 'constant']
ADD_CAP = [1]
#CLASSIFIER = [1]
#REBALANCES = [0, 1]
VAMP_MIX = [1]
DYNAMIC_BINARIZATION = [1]
PERM_RANGE = range(0, 5)

# dopnt iterate through them now
add_cap = ADD_CAP[0]
vamp_mix = VAMP_MIX[0]

for prior in PRIORS:
    if prior == 'vampprior_short':
        REBALANCES = [0, 1]
    elif prior == 'standard':
        REBALANCES = [0]

    for replay in REPLAYS:
        for rebalance in REBALANCES:  
            for db in DYNAMIC_BINARIZATION:
                for permid in PERM_RANGE:
                    command = "main_mnist.py \
                    --prior %(prior)s \
                    --replay_size %(replay)s \
                    --add_cap %(add_cap)s \
                    --use_vampmixingw %(vamp_mix)s \
                    --use_mixingw_correction %(rebalance)s \
                    --dynamic_binarization %(db)s \
                    --permindex %(permid)s" % locals()
            
                    print(command)
            
                    command = "{} cc_launch_cl.sh {}".format(sys.argv[1], command) 

                    os.system(command)
                    time.sleep(2)












