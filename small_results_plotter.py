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

    metric_plot_idxs = {'C-Index': (0,0),
                        'IBS': (0,1),
                        'MAE-Uncensored': (1,0),
                        'MAE-Hinge': (1,1)
                        }
    fig, ax = plt.subplots(2, 2, figsize=(15,13))
    for metric in METRICS.keys():
        metric_alt = METRICS[metric] #name of metric in results file
        metric_plot_idx = metric_plot_idxs[metric]
        # plt.subplots_adjust(right=0.77)
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
            ax[metric_plot_idx].plot(SEQS_LIST, means, label=alg, color=ALG_COLORS[alg], linestyle='dashed')
            ax[metric_plot_idx].errorbar(SEQS_LIST, means, yerr=stds, color=ALG_COLORS[alg], fmt="o", markersize=4, linewidth=2, capsize=4, capthick=2, alpha=0.5)
        ax[metric_plot_idx].set_xlabel('# Training Sequences', fontsize=15)
        ax[metric_plot_idx].set_ylabel(metric, fontsize=15)
        ax[metric_plot_idx].set_xticks([50, 100, 150, 200])
        ax[metric_plot_idx].tick_params(axis='x', labelsize=15)
        ax[metric_plot_idx].tick_params(axis='y', labelsize=15)
        # ax[metric_plot_idx].set_yticks(fontsize=20)
        ax[metric_plot_idx].grid()
        # plt
        # plt.title(f'{args.name}', fontsize=30)
        # plt.show()
    ax[0,0].legend(loc='lower right', fontsize=15)
    fig.suptitle(args.name, fontsize=30)
    plt.show()