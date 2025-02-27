import argparse
import json
import matplotlib.pyplot as plt
from pylab import rcParams
import numpy as np

NUM_TRIALS = 5
SEQS_LIST = [10, 25, 50, 100, 150, 200]
ALGS = {
        'Landmarking': 'SA', 
        'TCSR': 'LambdaSA', 
        'DeepTCSR': 'DeepLambdaSA', 
        'TC-MTLR': 'TC_MTLR', 
        'MTLR': 'MTLR'
        }
ALG_COLORS = {
        'Landmarking': 'red', 
        'TCSR': 'blue', 
        'DeepTCSR': 'purple', 
        'TC-MTLR': 'orange', 
        'MTLR': 'green'
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
    parser.add_argument('--name', type=str,
                        help='Name of dataset listed on plots')
    args = parser.parse_args()

    with open(args.path) as f:
        results_dict = json.load(f)

    #prints out hyperparameters
    for alg in ALGS.keys():
        alg_alt =  ALGS[alg] #name of algorithm in results file
        print(f'{alg} Hyperparameters: {results_dict["10"]["0"][alg_alt]["hyperparam_names"]}')
        for seq in SEQS_LIST:
            for trial in range(NUM_TRIALS):
                print(f'\tSeq: {seq}, Trial: {trial}, Values: {results_dict[str(seq)][str(trial)][alg_alt]["hyperparam_vals"]}')
    
    for metric in METRICS.keys():
        metric_alt = METRICS[metric] #name of metric in results file
        plt.figure(figsize=(15, 10.5))
        plt.subplots_adjust(right=0.77)
        for alg in ALGS.keys():
            alg_alt =  ALGS[alg] #name of algorithm in results file
            alg_results = [[] for _ in range(len(SEQS_LIST))]
            for seq_idx, seq in enumerate(SEQS_LIST):
                for trial in range(NUM_TRIALS):
                    metric_val = results_dict[str(seq)][str(trial)][alg_alt][metric_alt]
                    alg_results[seq_idx].append(metric_val)
            #print(alg_results)
            alg_results = np.array(alg_results)
            means = np.mean(alg_results, axis=1)
            #print(means)
            stds = np.std(alg_results, axis=1)
            plt.plot(SEQS_LIST, means, label=alg, color=ALG_COLORS[alg], linestyle='dashed')
            plt.errorbar(SEQS_LIST, means, yerr=stds, color=ALG_COLORS[alg], fmt="o", markersize=4, linewidth=2, capsize=4, capthick=2, alpha=0.5)
        plt.xlabel('# Training Sequences', fontsize=30)
        plt.ylabel(metric, fontsize=30)
        plt.xticks([50, 100, 150, 200], fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid()
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=22)
        plt.title(f'{args.name}', fontsize=30)
        plt.show()