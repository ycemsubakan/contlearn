#python main_mnist.py --dataset_name omniglot --prior vampprior_short --replay_size increase --use_mixingw_correction 0 --add_cap 1 --number_components 20 --number_components_init 20 --replay_type replay --notes 'debugging' --use_vampmixingw 1 --separate_means 0 --restart_means 1 --use_classifier 1  --use_replaycostcorrection 0 --use_visdom 1 --dynamic_binarization 1 --permindex 0 --semi_sup 0 --classifier_EP 150
#python main_mnist.py --dataset_name omniglot_char --prior vampprior_short --replay_size increase --use_mixingw_correction 0 --add_cap 1 --number_components 2 --number_components_init 2 --replay_type replay --notes 'debugging1' --use_vampmixingw 1 --separate_means 0 --restart_means 1 --use_classifier 1  --use_replaycostcorrection 0 --use_visdom 1 --dynamic_binarization 1 --permindex 0 --semi_sup 0 --classifier_EP 200 --use_entrmax 1 --epochs 2000 --visdom_server http://cem@nmf.cs.illinois.edu --visdom_env cem_dev2 --visdom_port 5800 --early_stopping_epochs 600 --classifier_rejection 0 
#python main_mnist.py --dataset_name omniglot --prior vampprior_short --replay_size increase --use_mixingw_correction 0 --add_cap 1 --number_components 50 --number_components_init 50 --replay_type replay --notes zhepei_exploring --use_vampmixingw 1 --separate_means 0 --restart_means 1 --use_classifier 1  --use_replaycostcorrection 0 --use_visdom 0 --dynamic_binarization 1 --permindex 0 --semi_sup 0 --classifier_EP 2 --use_entrmax 1 --epochs 200  --debug --early_stopping_epochs 50
python main_mnist.py --dataset_name dynamic_mnist --prior vampprior_short --replay_size increase --use_mixingw_correction 0 --add_cap 1 --number_components 2 --number_components_init 2 --replay_type replay --notes 'debugging1' --use_vampmixingw 1 --separate_means 0 --restart_means 1 --use_classifier 1  --use_replaycostcorrection 0 --use_visdom 1 --dynamic_binarization 1 --permindex 0 --semi_sup 0 --classifier_EP 20 --use_entrmax 1 --epochs 2 --visdom_server http://cem@nmf.cs.illinois.edu --visdom_env cem_dev2 --visdom_port 5800 --early_stopping_epochs 50 --classifier_rejection 0 --S 5

