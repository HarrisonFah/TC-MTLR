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
            'MAE-Hinge': 'mae_hinge',
            'MAE-PO': 'mae_po'
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

    metric_plot_idxs = {'C-Index': 0,
                        'IBS': 1,
                        'MAE-Uncensored': 2,
                        'MAE-Hinge': 3,
                        'MAE-PO': 4
                        }
    # fig, ax = plt.subplots(1, 5, figsize=(28, 6))

    axes = [
        plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2),
        plt.subplot2grid((2,6), (0,2), colspan=2),
        plt.subplot2grid((2,6), (0,4), colspan=2),
        plt.subplot2grid((2,6), (1,1), colspan=2),
        plt.subplot2grid((2,6), (1,3), colspan=2)
    ]

    fig = plt.gcf()
    fig.set_size_inches(17.5, 13.125)
    plt.subplots_adjust(hspace=0.25, wspace=0.75)

    for metric in METRICS.keys():
        metric_alt = METRICS[metric] #name of metric in results file
        metric_plot_idx = metric_plot_idxs[metric]
        for alg in ALGS.keys():
            alg_alt =  ALGS[alg] #name of algorithm in results file
            alg_results = [[] for _ in range(len(SEQS_LIST))]
            for seq_idx, seq in enumerate(SEQS_LIST):
                for trial in range(NUM_TRIALS):
                    metric_val = results_dict[str(seq)][str(trial)][alg_alt][metric_alt]
                    alg_results[seq_idx].append(metric_val)
            alg_results = np.array(alg_results)
            means = np.mean(alg_results, axis=1)
            stds = np.std(alg_results, axis=1)
            axes[metric_plot_idx].plot(SEQS_LIST, means, label=alg, color=ALG_COLORS[alg], linestyle='dashed')
            axes[metric_plot_idx].errorbar(SEQS_LIST, means, yerr=stds, color=ALG_COLORS[alg], fmt="o", markersize=4, linewidth=2, capsize=4, capthick=2, alpha=0.5)
        axes[metric_plot_idx].set_xlabel('# Training Sequences', fontsize=20)
        axes[metric_plot_idx].set_ylabel(metric, fontsize=20)
        axes[metric_plot_idx].set_xticks([50, 100, 150, 200])
        axes[metric_plot_idx].tick_params(axis='x', labelsize=15)
        axes[metric_plot_idx].tick_params(axis='y', labelsize=15)
        axes[metric_plot_idx].grid()
    axes[0].legend(loc='lower right', fontsize=15)
    # fig.suptitle(f'{args.name} Performance Results', fontsize=20)
    plt.suptitle(f'{args.name} Performance Results', fontsize=35)
    plt.show()