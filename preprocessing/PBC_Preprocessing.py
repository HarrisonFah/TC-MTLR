'''Builds a dictionary object for the PBC data with the following format:
{
    'seqs': FloatArray((num_seqs, horizon, features)), #sequence of states
    'cs': BoolArray((num_seqs,)), #censor indicator for each sequence
    'ts': IntArray((num_seqs,)), #absolute time of event for each sequence (in terms of # of states)
    'cols:': List(str), #names of features
    'rs': FloatArray((num_seqs, horizon)), #time between consecutive states in each sequence
    'seqs_ts': FloatArray((num_seqs, horizon)), #time until event for each state in each sequence
}'''

import math
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import itertools

input_path = '../data/DOC-10026794.dat'
output_path = '../data/PBC.pkl'

columns = ['case_num', 'time_of_event', 'status', 'drug', 'age', 'sex', 'day', 'ascites', 'hepatomegaly', 'spiders', 'edema', 'serBilir', 'serChol', 'albumin', 'alkaline', 'SGOT', 'platelets', 'prothrombin', 'histologic']

state_features = ['drug', 'age', 'sex', 'ascites', 'hepatomegaly', 'spiders', 'edema', 'serBilir', 'serChol', 'albumin', 'alkaline', 'SGOT', 'platelets', 'prothrombin', 'histologic']
continuous_features = ['age', 'serBilir', 'serChol', 'albumin', 'alkaline', 'SGOT', 'platelets', 'prothrombin'] #continuous features are normalized
discrete_features = ['drug', 'sex', 'ascites', 'hepatomegaly', 'spiders', 'edema', 'histologic']

def main():
    df = pd.read_table(input_path, header=None, delim_whitespace=True)
    df.columns = columns
    print(df.describe())

    #gets median of non-empty columns values to replace empty values
    df = df.replace(to_replace='.', value=-1)
    for column in df.columns:
        df[column] = df[column].astype(float)
        median = df[column].median()
        df[column] = df[column].replace(to_replace=-1, value=median)
        if column in continuous_features: #normalizes continuous features
            df[column]=(df[column]-df[column].mean())/df[column].std()

    all_rs = [] #used to get statistics of all consecutive time between states
    all_end_ts = [] #used to get statistics of time between final state and time of event

    seqs = []
    cs = []
    ts = []
    rs = []
    seqs_ts = []
    horizon = -1

    seq = []
    c = -1
    t = -1
    r = []
    seq_t = []

    prev_id = -1
    prev_day = 0
    #iterate through all rows to build sequence data
    for index, row in df.iterrows():
        current_id = int(row['case_num'])
        print(f"Index:{index}, ID:{current_id}")
        current_day = int(row['day'])
        #if reached a new case, save the previous sequence
        if index > 0 and current_id != prev_id:
            if len(seq) > 1:
                seqs.append(seq)
                cs.append(c)
                ts.append(len(seq))
                r.append(seq_t[0] - prev_day)
                all_end_ts.append(seq_t[0] - prev_day)
                rs.append(r)
                seqs_ts.append(seq_t)
                if len(seq) > horizon:
                    horizon = len(seq)

            seq = []
            c = -1
            t = -1
            r = []
            seq_t = []
            prev_day = 0
        state = row[state_features].tolist()
        seq.append(state)
        #if patient is alive or transplanted, they are censored
        if int(row['status']) in [0,1]: 
            c = True
        else:
            c = False
        if current_day > 0:
            r.append(current_day - prev_day)
            all_rs.append(current_day - prev_day)
        seq_t.append(int(row['time_of_event']) - current_day)
        prev_id = current_id
        prev_day = current_day
    #add data from final sequence
    seqs.append(seq)
    cs.append(c)
    ts.append(len(seq)-1)
    r.append(seq_t[0] - prev_day)
    all_end_ts.append(seq_t[0] - prev_day)
    rs.append(r)
    seqs_ts.append(seq_t)

    #pads arrays with zeros so they all have same length
    for seq_idx in range(len(seqs)):
        missing_states = horizon-len(seqs[seq_idx])
        rs[seq_idx] += [0 for state in range(missing_states)]
        seqs_ts[seq_idx] += [0 for state in range(missing_states)]
        seqs[seq_idx] = seqs[seq_idx] + [[0 for feature in range(len(state_features))] for state in range(missing_states)]
    pbc_dict = {
        'seqs': np.array(seqs),
        'cs': np.array(cs),
        'ts': np.array(ts),
        'cols': state_features,
        'rs': np.array(rs),
        'seqs_ts': np.array(seqs_ts)
    }
    with open(output_path, 'wb') as f:
        pickle.dump(pbc_dict, f) 

    #compute statistics of time between states
    print("Statistics of Time Between States:")
    print(f"\tMinimum: {np.min(all_rs)}")
    print(f"\tMaximum: {np.max(all_rs)}")
    print(f"\tMedian: {np.median(all_rs)}")
    print(f"\tMean: {np.mean(all_rs)}")
    print(f"\tStd. Dev.: {np.std(all_rs)}")

    #compute statistics of time between last state and time of event
    print("Statistics of Time Last State and Time of Event:")
    print(f"\tMinimum: {np.min(all_end_ts)}")
    print(f"\tMaximum: {np.max(all_end_ts)}")
    print(f"\tMedian: {np.median(all_end_ts)}")
    print(f"\tMean: {np.mean(all_end_ts)}")
    print(f"\tStd. Dev.: {np.std(all_end_ts)}")

    print("Statistics:")
    print("\tDataset Size:", np.array(seqs).shape[0])
    print("\tHorizon:", np.array(seqs).shape[1])
    print("\t# Features:", np.array(seqs).shape[2])
    print("\tMedian Length:", np.median(ts))
    print("\tMean Length:", np.mean(ts))
    print("\tStd. Dev. Length:", np.std(ts))
    print("\t% Censored:", np.mean(cs)*100)

main()