import argparse
import json
import numpy as np
import scipy.stats as stats 
from itertools import combinations

ALGS = {
        'Landmarking': 'SA1', 
        'DeepTCSR (0)': 'DeepLambdaSA0', 
        'DeepTCSR (0.9)': 'DeepLambdaSA0.9', 
        'DeepTCSR (0.95)': 'DeepLambdaSA0.95', 
        'MTLR': 'MTLR0',
        'TC-MTLR (0)': 'TC_MTLR0',
        'TC-MTLR (0.9)': 'TC_MTLR0.9',
        'TC-MTLR (0.95)': 'TC_MTLR0.95',
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
    parser.add_argument('--key', type=str, default='null',
                        help='Top level key used in results dictionary.')
    parser.add_argument('--num_trials', type=int, default=5,
                        help='Number of trials to evaluate over.')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Alpha value used for p-value.')
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
        print("\tMean (Std. Dev)")
        metric_alt = METRICS[metric] #name of metric in results file
        alg_results = {}
        for alg in ALGS.keys():
            alg_alt =  ALGS[alg] #name of algorithm in results file
            alg_results[alg] = []
            for trial in range(args.num_trials):
                metric_val = results_dict[args.key][str(trial)][alg_alt][metric_alt]
                alg_results[alg].append(metric_val)
            alg_results_np = np.array(alg_results[alg])
            mean = np.mean(alg_results_np)
            std = np.std(alg_results_np)
            print(f"\t\t{alg}: {mean} ({std})")
        print("\tMean Difference (Std. Dev), P-Value")
        alg_pairs = list(combinations(ALGS.keys(), 2))
        for pair in alg_pairs:
            alg_1 = pair[0]
            alg_1_results = np.array(alg_results[alg_1])
            alg_2 = pair[1]
            alg_2_results = np.array(alg_results[alg_2])
            p_value = stats.ttest_rel(alg_1_results, alg_2_results).pvalue
            if p_value < 0.05:
                print(f"\t\t{alg_1} vs {alg_2}: {np.mean(alg_1_results - alg_2_results)} ({np.std(alg_1_results - alg_2_results)}), {p_value}")
        
        