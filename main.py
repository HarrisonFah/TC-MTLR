import os
from shutil import copyfile

import yaml
import argparse
import gc
import itertools

from lambda_cox import LambdaSA
from baseline_cox import SA
from deep_lambda_cox import DeepLambdaSA
from tc-mtlr import TC_MTLR
from utils import get_train_val_test_mtlr


os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"  # see XXXX

def hypertune(agent_obj, hyperparams, data_arrays, time_bins, mtlr_gens, config):
    hyperparam_groups = list(itertools.product(*hyperparams.values))
    top_cindex = -1
    top_hyperparams = None
    top_agent = None
    top_gens = None
    for hyperparam_vals in hyperparam_groups:
        for hyper_idx, hyperparam_name in hyperparams.names:
            config[hyperparam_name] = hyperparam_vals[hyper_idx]
        agent = agent_obj(config)
        train_gen, val_gen, test_gen = agent.get_train_val_test(data_arrays)
        agent.train(train_gen)
        train_gen_mtlr, val_gen_mtlr, test_gen_mtlr = mtlr_gens
        cindex, _, _, _, _ = agent.evaluate(train_gen_mtlr, val_gen_mtlr, time_bins)
        if cindex > top_cindex:
            top_cindex = cindex
            top_hyperparams = hyperparam_vals
            top_agent = agent
            top_gens = (train_gen, val_gen, test_gen)
    return top_cindex, top_hyperparams, top_agent, top_gens

if __name__ == '__main__':  

    #Read config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, 
                        help='Path to configuration file.')
    parser.add_argument('--exp_name', type=str, default='exp_1', 
                        help='exp name')
    parser.add_argument('--seed', help='Experiment seed', type=int, default=42)
    parser.add_argument('--val_size', help='Validation set ratio', type=float, default=.15)
    parser.add_argument('--test_size', help='Test set ratio', type=float, default=.2)
    args = parser.parse_args()

    config_path = args.config
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    exp_name = args.exp_name
    seed = args.seed
    val_size = args.val_size
    test_size = args.test_size

    output_file = config['output_file'].format(args.agent)
    output_file = os.path.join(output_file, 
                               config['dataset_name'],
                               exp_name)
    config['output_file'] = output_file

    # dataset info
    seqs, ts, cs, h_tgt, h_ws, mask = get_data(config.dataset_name,
                                                config.landmark,
                                                config.calculate_tgt_and_mask,
                                                config.dataset_kwargs)
    seqs = seqs.astype(np.float32)
    data = {'seqs': seqs,
            'ts': ts,
            'cs': cs,
            'h_ws': h_ws,
            'target': h_tgt,
            'mask': mask}

    data_arrays = train_val_test_split(data['seqs'],
                                    data['target'],
                                    data['h_ws'],
                                    data['mask'],
                                    data['ts'],
                                    data['cs'],
                                    data['rs'],
                                    data['seqs_ts'],
                                    seed=seed,
                                    val_size=val_size,
                                    test_size=test_size)

    config['input_dim'] = data_arrays[0].shape[2]

    mtlr_gens = get_train_val_test_mtlr(data_arrays)

    time_bins = mean_time_bins(data_arrays, config.dataset_kwargs['horizon'])
    config['time_bins'] = time_bins

    for type_agent in ['SA_Init_Cox', 'SA_Landmark_Cox', 'LambdaSA_Cox', 'DeepLambdaSA_Cox', 'TC_MTLR']:
        if type_agent == 'SA_Init_Cox'
            agent_obj = SA
            config['landmark'] = False 
        elif type_agent == 'SA_Landmark_Cox':
            agent_obj = SA
            config['landmark'] = True 
        elif type_agent == 'LambdaSA_Cox':
            agent_obj = LambdaSA
        elif type_agent == 'DeepLambdaSA_Cox':
            agent_obj = DeepLambdaSA
        elif type_agent == 'TC_MTLR':
            agent_obj = TC_MTLR
        hyperparams = config[type_agent].hyperparams

        try:
            cindex, hyperparams, agent, gens = hypertune(agent_obj, hyperparams, data_arrays, time_bins, mtlr_gens, config)
            train_gen, val_gen, test_gen = mtlr_gens
            cindex, ibs, mae_uncensored, mae_hinge, mae_po = agent.evaluate(train_gen, test_gen, time_bins)
            path_result = os.path.join(output_file, f'{type_agent}_results.json')
            data = {'cindex':cindex, 'ibs':ibs, 'mae_uncensored':mae_uncensored, 'mae_hinge':mae_hinge, 'mae_po':mae_po}
            with open(path_result, 'w') as json_file:
                json.dump(data, json_file)
            
        except KeyboardInterrupt:
            gc.collect()
            pass
