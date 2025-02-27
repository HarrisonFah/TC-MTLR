import argparse
import json
import matplotlib.pyplot as plt
from pylab import rcParams
import numpy as np

NUM_TRIALS = 1

# ALGS = {
#         'Landmarking': 'SA', 
#         'DeepTCSR': 'DeepLambdaSA', 
#         'TC-MTLR': 'TC_MTLR', 
#         'MTLR': 'MTLR'
#         }
ALGS = {
        'DeepTCSR': 'DeepLambdaSA',
        }
METRICS = {
            'C-Index': 'cindex',
            'IBS': 'ibs', 
            'MAE-Uncensored': 'mae_uncensored',
            'MAE-Hinge': 'mae_hinge'
            }

if __name__ == '__main__':  

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, 
                        help='Path to json file.')
    parser.add_argument('--key', type=str,
                        help='Top level key used in results dictionary.')
    parser.add_argument('--num_trials', type=int,
                        help='Number of trials to evaluate over.')
    args = parser.parse_args()

    with open(args.path) as f:
        results_dict = json.load(f)

    #prints out hyperparameters
    for alg in ALGS.keys():
        alg_alt =  ALGS[alg] #name of algorithm in results file
        print(f'{alg} Hyperparameters: {results_dict[args.key]["0"][alg_alt]["hyperparam_names"]}')
        for trial in range(args.num_trials):
            print(f'\tTrial: {trial}, Values: {results_dict[args.key][str(trial)][alg_alt]["hyperparam_vals"]}')
    
    for metric in METRICS.keys():
        print("Metric:", metric)
        metric_alt = METRICS[metric] #name of metric in results file
        for alg in ALGS.keys():
            alg_alt =  ALGS[alg] #name of algorithm in results file
            alg_results = []
            for trial in range(args.num_trials):
                metric_val = results_dict[args.key][str(trial)][alg_alt][metric_alt]
                alg_results.append(metric_val)
            alg_results = np.array(alg_results)
            mean = np.mean(alg_results)
            std = np.std(alg_results)
            print(f"{alg}: {mean} ({std})")
        