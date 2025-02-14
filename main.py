import os
from shutil import copyfile


import yaml
import argparse
import gc

from lambda_cox import LambdaSA
from baseline_cox import SA
from deep_lambda_cox import DeepLambdaSA
from tc_mtlr import TC_MTLR
from utils import median_time_bins, quantile_time_bins


os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"  # see XXXX

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

    output_file = config['output_file'].format(args.agent)
    output_file = os.path.join(output_file, 
                               config['dataset_name'],
                               config['arch']['type'],
                               'lambda_{}'.format(config['lambda_']),
                               'landmark_{}'.format(config['landmark']),
                               exp_name)
    config['output_file'] = output_file

    lambda_cox=False
    if type_agent == "SA":
        agent = SA(config, seed)
    elif type_agent == 'LambdaSA':
        agent = LambdaSA(config, seed)
        lambda_cox=True
    elif type_agent == 'DeepLambdaSA':
        agent = DeepLambdaSA(config, seed)
    elif type_agent == 'TC_MTLR':
        config['layer_size'] = 16
        config['num_hidden'] = 1
        agent = TC_MTLR(config, seed)
    else:
        raise Exception('Agent type not found')

    try:
        # import ipdb; ipdb.set_trace()
        if type_agent == 'LambdaSA':
            X_train, ts_train, cs_train, train_gen, val_gen, test_gen = agent.get_train_val_test(test_size=size)
            agent.train(X_train, ts_train, cs_train)
        else:
            train_gen, val_gen, test_gen = agent.get_train_val_test(test_size=size)
            agent.train(train_gen)
        #time_bins = median_time_bins(train_gen, agent.horizon)
        #time_bins = quantile_time_bins(train_gen, agent.horizon)
        #train_gen, test_gen = agent.get_train_test(test_size=size)
        #agent.save()
        #agent.eval(test_gen)
        isds, cindex, ibs, mae_uncensored, mae_hinge, mae_po = agent.eval(train_gen, test_gen, agent.time_bins, lambda_cox)
        print(cindex, ibs, mae_uncensored, mae_hinge, mae_po)
        # config_path = os.path.join(output_file, 'config.yaml')
        # with open(config_path, 'w') as outfile:
        #     yaml.dump(config, outfile, default_flow_style=False)
        
    except KeyboardInterrupt:
        gc.collect()
        pass
