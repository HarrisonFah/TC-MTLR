import os
from shutil import copyfile
import json

import yaml
import argparse
import gc
import itertools
from numpy import random
import time

from lambda_cox import LambdaSA
from baseline_cox import SA
from deep_lambda_cox import DeepLambdaSA
from tc_mtlr import TC_MTLR
from mtlr import MTLR
from utils import median_time_bins, quantile_time_bins

if __name__ == '__main__':  

    #Read config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, 
                        help='Path to configuration file.')
    parser.add_argument('--exp_name', type=str, default='exp_1', 
                        help='exp name')
    parser.add_argument('--agent', type=str, default='SA',
                        help='SA or LambdaSA or DeepLambdaSA')
    parser.add_argument('--seed', help='Experiment seed', type=int, default=42)
    parser.add_argument('--size', help='Test set ratio', type=float, default=.2)

    # Overwrites some entries in config
    parser.add_argument('--taskid', help='Task id', type=int, default=None)
    parser.add_argument('--lambda_', help='Lambda', type=float, default=None)
    parser.add_argument('--landmark', help='Landmarking', type=int, default=None)
    args = parser.parse_args()

    config_path = args.config
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    exp_name = args.exp_name
    seed = args.seed
    type_agent = args.agent
    size = args.size

    #Model parameters
    if args.taskid is not None:
        config['dataset_kwargs']['task_id'] = args.taskid

    if args.lambda_ is not None:
        config['lambda_'] = args.lambda_
    
    if args.landmark is not None:
        config['landmark'] = bool(args.landmark)

    output_file = config['output_file'].format(config['dataset_name'])
    if not os.path.exists(output_file):
        os.makedirs(output_file)
    output_file = os.path.join(output_file, 'results.json')
    config['output_file'] = output_file
    if len(config['num_seqs']) > 0:
        num_seqs_list = config['num_seqs']
    else:
        num_seqs_list = [None]

    try:
        results_dict = {}
        trial_seeds = random.randint(1, 1000, size=(config['num_trials'],)) #generate seeds so each trial across training sizes includes samples from previous training sizes
        for num_seqs in num_seqs_list:
            print(f'Number of Sequences: {num_seqs}')
            results_dict[num_seqs] = {}
            for trial in range(config['num_trials']):
                seed = trial_seeds[trial]
                print(f'\tTrial: {trial}')
                results_dict[num_seqs][trial] = {}
                for type_agent in ["SA", "LambdaSA", "DeepLambdaSA", "TC_MTLR", "MTLR"]:
                    print(f'\t\tAgent: {type_agent}')
                    train_gen = None
                    val_gen = None
                    test_gen = None
                    X_train = None
                    ts_train = None
                    cs_train = None

                    top_cindex = -1
                    top_mae = float('inf')
                    top_hyperparams = None
                    top_agent = None
                    top_time_bins = None

                    hyperparams = config['hyperparams'][type_agent]
                    hyperparam_groups = list(itertools.product(*hyperparams['vals']))

                    config['axis'] = 2
                    config['use_quantiles'] = False

                    for group_idx, hyperparam_vals in enumerate(hyperparam_groups):
                        print(f'\t\t\tHyperparam Index: {group_idx}')
                        for hyper_idx, hyperparam_name in enumerate(hyperparams['names']):
                            config[hyperparam_name] = hyperparam_vals[hyper_idx]

                        lambda_cox=False
                        if type_agent == "SA":
                            config['lambda_'] = 1
                            agent = DeepLambdaSA(config, seed)
                        elif type_agent == 'LambdaSA':
                            config['axis'] = 1
                            config['lambda_'] = 0
                            lambda_cox=True
                            agent = LambdaSA(config, seed)
                        elif type_agent == 'DeepLambdaSA':
                            agent = DeepLambdaSA(config, seed)
                        elif type_agent == 'TC_MTLR':
                            agent = TC_MTLR(config, seed)
                        elif type_agent == 'MTLR':
                            agent = MTLR(config, seed)
                        else:
                            raise Exception('Agent type not found')

                        if type_agent == 'LambdaSA' and X_train is None:
                            X_train, ts_train, cs_train, train_gen, val_gen, test_gen = agent.get_train_val_test(test_size=size, num_train_seqs=num_seqs)
                        elif train_gen is None:
                            train_gen, val_gen, test_gen = agent.get_train_val_test(test_size=size, num_train_seqs=num_seqs)
                            state, next_state, reward, not_done, times, censors = train_gen.get_all_data()

                        if type_agent in ['SA', 'LambdaSA', 'DeepLambdaSA']:
                            agent.set_time_bins(train_gen, val_gen, test_gen)
                        else:
                            agent.init_networks(train_gen, val_gen, test_gen)
                        
                        start_time = time.time()
                        if type_agent == 'LambdaSA':
                            agent.train(X_train, ts_train, cs_train)
                        else:
                            agent.train(train_gen)
                        end_time = time.time()

                        isds, cindex, ibs, mae_uncensored, mae_hinge, maepo = agent.eval(train_gen, val_gen, agent.time_bins, lambda_cox)
                        if cindex > top_cindex:
                            top_cindex = cindex
                            top_hyperparams = hyperparam_vals
                            top_agent = agent
                            top_time_bins = agent.time_bins

                    isds, cindex, ibs, mae_uncensored, mae_hinge, maepo = top_agent.eval(train_gen, test_gen, top_time_bins, lambda_cox)
                    results_dict[num_seqs][trial][type_agent] = {
                                                'hyperparam_names': hyperparams['names'],
                                                'hyperparam_vals': top_hyperparams,
                                                'cindex': cindex,
                                                'ibs': ibs,
                                                'mae_uncensored': mae_uncensored,
                                                'mae_hinge': mae_hinge,
                                                'mae_po': maepo
                                                }
                    with open(output_file, 'w') as out:
                        json.dump(results_dict, out, indent=4)
                        
    except KeyboardInterrupt:
        gc.collect()
        pass
