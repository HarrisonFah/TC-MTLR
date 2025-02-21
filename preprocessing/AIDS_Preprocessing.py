'''Builds a dictionary object for the AIDS data with the following format:
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

input_path = '../data/aids.csv'
output_path = '../data/AIDS.pkl'

columns = ['patient', 'Time', 'death', 'CD4', 'obstime', 'drug', 'gender', 'prevOI', 'AZT', 'start', 'stop', 'event']

state_features = ['CD4', 'drug', 'gender', 'prevOI', 'AZT']
continuous_features = ['CD4'] #continuous features are normalized
discrete_features = ['drug', 'gender', 'prevOI', 'AZT']
feature_encodings = {
    'drug': {
        'ddC': 0,
        'ddI': 1,
    },
    'gender': {
        'male': 0,
        'female': 1,
    },
    'prevOI': {
        'noAIDS': 0,
        'AIDS': 1,
    },
    'AZT': {
        'failure': 0,
        'intolerance': 1,
    }
}

def main():
    df = pd.read_csv(input_path)
    df.drop(columns='Unnamed: 0',inplace=True)
    print(df.describe())

    #encode discrete features and normalize continuous features
    for column in df.columns:
        if column in continuous_features:
            df[column]=(df[column]-df[column].mean())/df[column].std()
        elif column in discrete_features:
            df[column] = df[column].replace(to_replace=feature_encodings[column])

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
    prev_start = -1
    prev_end = -1
    #iterate through all rows to build sequence data
    for index, row in df.iterrows():
        current_id = int(row['patient'])
        print(f"Index:{index}, ID:{current_id}")
        start_time = int(row['start'])
        end_time = int(row['stop'])
        #if reached a new case, save the previous sequence
        if index > 0 and current_id != prev_id:
            if len(seq) > 1:
                seqs.append(seq)
                cs.append(c)
                ts.append(len(seq))
                rs.append(r)
                seqs_ts.append(seq_t)
                all_rs.pop() #remove terminal states
                all_end_ts.append(prev_end - prev_start)
                if len(seq) > horizon:
                    horizon = len(seq)
            seq = []
            c = -1
            t = -1
            r = []
            seq_t = []
        state = row[state_features].tolist()
        seq.append(state)
        #if patient is alive they are censored
        if int(row['death']) == 0: 
            c = True
        else:
            c = False
        r.append(end_time - start_time)
        all_rs.append(end_time - start_time)
        seq_t.append(int(row['Time']) - start_time)
        prev_id = current_id
        prev_start = start_time
        prev_end = int(row['Time'])
    #add data from final sequence
    seqs.append(seq)
    cs.append(c)
    ts.append(len(seq)-1)
    all_rs.pop() #remove terminal states
    all_end_ts.append(end_time - start_time)
    rs.append(r)
    seqs_ts.append(seq_t)

    #pads arrays with zeros so they all have same length
    for seq_idx in range(len(seqs)):
        missing_states = horizon-len(seqs[seq_idx])
        rs[seq_idx] += [0 for state in range(missing_states)]
        seqs_ts[seq_idx] += [0 for state in range(missing_states)]
        seqs[seq_idx] = seqs[seq_idx] + [[0 for feature in range(len(state_features))] for state in range(missing_states)]
    aids_dict = {
        'seqs': np.array(seqs),
        'cs': np.array(cs),
        'ts': np.array(ts),
        'cols': state_features,
        'rs': np.array(rs),
        'seqs_ts': np.array(seqs_ts)
    }
    with open(output_path, 'wb') as f:
        pickle.dump(aids_dict, f) 

    print("Horizon:", horizon)

    #compute statistics of time between states
    print("\nStatistics of Time Between States:")
    print(f"\tMinimum: {np.min(all_rs)}")
    print(f"\tMaximum: {np.max(all_rs)}")
    print(f"\tMedian: {np.median(all_rs)}")
    print(f"\tMean: {np.mean(all_rs)}")
    print(f"\tStd. Dev.: {np.std(all_rs)}")

    #compute statistics of time between last state and time of event
    print("\nStatistics of Time Last State and Time of Event:")
    print(f"\tMinimum: {np.min(all_end_ts)}")
    print(f"\tMaximum: {np.max(all_end_ts)}")
    print(f"\tMedian: {np.median(all_end_ts)}")
    print(f"\tMean: {np.mean(all_end_ts)}")
    print(f"\tStd. Dev.: {np.std(all_end_ts)}")

main()