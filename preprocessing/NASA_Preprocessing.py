'''Builds a dictionary object for the AIDS data with the following format:
{
    'seqs': FloatArray((num_seqs, horizon, features)), #sequence of states
    'cs': BoolArray((num_seqs,)), #censor indicator for each sequence
    'ts': IntArray((num_seqs,)), #absolute time of event for each sequence (in terms of # of states)
    'cols:': List(str), #names of features
    'rs': FloatArray((num_seqs, horizon)), #time between consecutive states in each sequence
    'seqs_ts': FloatArray((num_seqs, horizon)), #time until event for each state in each sequence
}'''

#Based on code notebook from: https://www.kaggle.com/code/vinayak123tyagi/damage-propagation-modeling-for-aircraft-engine/notebook

import math
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import itertools

train_input_path = '../data/train_FD001.txt'
test_input_path = '../data/test_FD001.txt'
output_path = '../data/NASA.pkl'

columns = ['unit_number','time_in_cycles','setting_1','setting_2','TRA','T2','T24','T30','T50','P2','P15','P30','Nf',
           'Nc','epr','Ps30','phi','NRf','NRc','BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32']

state_features = ['setting_1','setting_2','T24','T30','T50','P15','P30','Nf',
                'Nc','Ps30','phi','NRf','NRc','BPR','htBleed','W31','W32']

#function for preparing training data and forming a RUL column with information about the remaining
# before breaking cycles
def prepare_data(data, factor = 0):
    df = data.copy()
    fd_RUL = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    fd_RUL = pd.DataFrame(fd_RUL)
    fd_RUL.columns = ['unit_number','max']
    df = df.merge(fd_RUL, on=['unit_number'], how='left')
    df['RUL'] = df['max'] - df['time_in_cycles']
    df.drop(columns=['max'],inplace = True)
    
    return df[df['time_in_cycles'] > factor]

def main():

    train_df = pd.read_csv(train_input_path, sep=" ", header=None)
    train_df.drop(columns=[26,27],inplace=True)
    train_df.columns = columns
    train_df = train_df.assign(censored = [False for row in range(len(train_df.index))])
    test_df = pd.read_csv(test_input_path, sep=" ", header=None)
    test_df.drop(columns=[26,27],inplace=True)
    test_df.columns = columns
    test_df = test_df.assign(censored = [True for row in range(len(test_df.index))])
    df = pd.concat([train_df, test_df])
    #delete columns with constant values ​​that do not carry information about the state of the unit
    df.drop(columns=['Nf_dmd','PCNfR_dmd','P2','T2','TRA','farB','epr'],inplace=True)
    df = prepare_data(df)
    print(df.describe())

    #normalize continuous features
    for column in df.columns:
        if column in state_features:
            df[column]=(df[column]-df[column].mean())/df[column].std()

    horizon = int(df['time_in_cycles'].max())

    seqs = []
    cs = []
    ts = []
    rs = []
    seqs_ts = []

    seq = []
    c = -1
    t = -1
    r = []
    seq_t = []

    prev_id = -1
    prev_start = -1
    prev_end = -1
    censor_count = 0
    uncensor_count = 0
    #iterate through all rows to build sequence data
    for index, row in df.iterrows():
        current_id = int(row['unit_number'])
        #if reached a new case, save the previous sequence
        if index > 0 and current_id != prev_id:
            print(f"Index:{index}, ID:{current_id}")
            seqs.append(seq)
            cs.append(c)
            ts.append(len(seq))
            rs.append(r)
            seq_t = [i for i in range(len(seq)-1, -1, -1)]
            seqs_ts.append(seq_t)
            seq = []
            c = -1
            t = -1
            r = []
            seq_t = []
        state = row[state_features].tolist()
        seq.append(state)
        r.append(1)
        #seq_t.append(int(row['RUL']))
        prev_id = current_id
        prev_time = int(row['time_in_cycles'])
        c = row['censored']
    #add data from final sequence
    seqs.append(seq)
    cs.append(c)
    ts.append(len(seq)-1)
    rs.append(r)
    seq_t = [i for i in range(len(seq)-1, -1, -1)]
    seqs_ts.append(seq_t)

    #pads arrays with zeros so they all have same length
    for seq_idx in range(len(seqs)):
        missing_states = horizon-len(seqs[seq_idx])
        rs[seq_idx] += [0 for state in range(missing_states)]
        seqs_ts[seq_idx] += [0 for state in range(missing_states)]
        seqs[seq_idx] = seqs[seq_idx] + [[0 for feature in range(len(state_features))] for state in range(missing_states)]

    nasa_dict = {
        'seqs': np.array(seqs),
        'cs': np.array(cs),
        'ts': np.array(ts),
        'cols': state_features,
        'rs': np.array(rs),
        'seqs_ts': np.array(seqs_ts)
    }
    with open(output_path, 'wb') as f:
        pickle.dump(nasa_dict, f)

    print("Statistics:")
    print("\tDataset Size:", np.array(seqs).shape[0])
    print("\tHorizon:", np.array(seqs).shape[1])
    print("\t# Features:", np.array(seqs).shape[2])
    print("\tMedian Length:", np.median(ts))
    print("\tMean Length:", np.mean(ts))
    print("\tStd. Dev. Length:", np.std(ts))
    print("\t% Censored:", np.mean(cs)*100)


main()