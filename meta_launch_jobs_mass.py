import numpy as np
import pdb
import os
import time
import sys

try:
    host = sys.argv[2]
except: 
    host = 'cedar'
print(host)

RESULT_DIR = 'real_result_dir'


PRIORS = ['vampprior_short', 'standard']
REPLAYS = ['increase', 'constant']
ADD_CAP = [1]
VAMP_MIX = [1]
DYNAMIC_BINARIZATION = [1]
LAMBDA = [5]

if host == 'cedar':
    PERM_RANGE = range(1, 5)
else: 
    PERM_RANGE = range(5, 10)
    
#!!!!!1
#PERM_RANGE = [0]

# dopnt iterate through them now
add_cap = ADD_CAP[0]
vamp_mix = VAMP_MIX[0]
Lambda = LAMBDA[0]


for prior in PRIORS:
    if prior == 'vampprior_short':
        REBALANCES = [0, 1]
    elif prior == 'standard':
        REBALANCES = [0]

    for replay in REPLAYS:
        for rebalance in REBALANCES:  
            for db in DYNAMIC_BINARIZATION:
                for permid in PERM_RANGE:

                    if (prior == 'vampprior_short') and replay == 'constant':
                        COSTC = [0, 1]
                    else: 
                        COSTC = [1]

                    for costc in COSTC:
                        command = "main_mnist.py \
                        --prior %(prior)s \
                        --replay_size %(replay)s \
                        --add_cap %(add_cap)s \
                        --use_vampmixingw %(vamp_mix)s \
                        --use_mixingw_correction %(rebalance)s \
                        --use_classifier 0 \
                        --semi_sup 1 \
                        --dynamic_binarization %(db)s \
                        --permindex %(permid)s \
                        --use_replaycostcorrection %(costc)s \
                        --Lambda %(Lambda)s \
                        --notes llcorb2" % locals()
                
                        print(command)
                
                        if host == 'cedar':
                            command = "{} cc_launch_cl.sh {}".format(sys.argv[1], command) 
                        else:
                            command = "{} cc_launch_cl_graham.sh {}".format(sys.argv[1], command) 

                        os.system(command)
                        time.sleep(2)

