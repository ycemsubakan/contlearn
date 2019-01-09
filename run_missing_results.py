import pickle
import pdb
import os
import copy
import sys
import time


try:
    host = sys.argv[2]
except: 
    host = 'cedar'
print(host)

path = 'results_files'
files = os.listdir(path)

filesf = []

PRIORS = ['vampprior_short', 'standard']
REPLAYS = ['increase', 'constant']
add_cap = 1
vamp_mix = 1
DYNAMIC_BINARIZATION = [1]
PERM_RANGE = range(10, 15)

all_combs = 0 
found_files = []
missing_files = []
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

                        all_combs = all_combs + 1
                        found = 0 
                        for fl in files: 
                            if (prior in fl) and (('replay_size_' + replay) in fl) and (('mixingw_correction_' + str(rebalance) ) in fl) \
                                    and (('permutation_' + str(permid))  in fl) and (('replaycostcorrection_' + str(costc) in fl)):
                                    
                                    found_files.append(fl)                          
                                    found = 1
                                    break

                        if (found == 0): 
                            mf = prior + 'replay_size_' + replay + 'mixingw_correction_' + str(rebalance) + 'permutation_' + str(permid) + 'replaycostcorrection_' + str(costc)
                            print(mf + '\n')
                            missing_files.append(mf)

                            command = "main_mnist.py \
                            --prior %(prior)s \
                            --replay_size %(replay)s \
                            --add_cap %(add_cap)s \
                            --use_vampmixingw %(vamp_mix)s \
                            --use_mixingw_correction %(rebalance)s \
                            --dynamic_binarization %(db)s \
                            --permindex %(permid)s \
                            --use_replaycostcorrection %(costc)s" % locals()
                    
                            #print(command)
                    
                            if host == 'cedar':
                                command = "{} cc_launch_cl.sh {}".format(sys.argv[1], command) 
                            else:
                                command = "{} cc_launch_cl_graham.sh {}".format(sys.argv[1], command) 

                            os.system(command)
                            time.sleep(2)

#print(found_files[:4])
print('found files {}'.format(len(found_files)))
print('all combinations {}'.format(all_combs))

