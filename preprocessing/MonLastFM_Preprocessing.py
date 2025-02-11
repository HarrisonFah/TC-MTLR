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
import sklearn
sklearn.set_config(transform_output="pandas")
import category_encoders as ce

input_path = '../data/lastfm-dataset-1k/userid-timestamp-artid-artname-traid-traname.tsv'
output_path = '../data/MonLastFM.pkl'

columns = ['userid', 'timestamp', 'musicbrainz-artist-id', 'artist-name', 'musicbrainz-track-id', 'track-name']

state_features = ['musicbrainz-artist-id', 'musicbrainz-track-id',]

def main():
    df = pd.read_csv(input_path, sep='\t', header=None, on_bad_lines='skip')
    df.columns = columns
    print(df.describe())

    n_unique_artist = df['musicbrainz-artist-id'].nunique()
    artist_encoder = ce.BaseNEncoder(base=n_unique_artist)
    df['musicbrainz-artist-id'] = artist_encoder.fit_transform(df['musicbrainz-artist-id'])

    n_unique_song = df['musicbrainz-track-id'].nunique()
    song_encoder = ce.BaseNEncoder(base=n_unique_song)
    df['musicbrainz-track-id'] = artist_encoder.fit_transform(df['musicbrainz-track-id'])

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
    prev_month = 0
    prev_year = 0
    prev_state = 0
    #iterate through all rows to build sequence data
    for index, row in df.iterrows():
        current_id = int(row['userid'])
        print(f"Index:{index}, ID:{current_id}")
        current_date = datetime.datetime.fromisoformat(row['timestamp'])
        current_month = current_data.month
        current_year = current_data.year
        #if reached a new case, save the previous sequence
        if index > 0 and current_id != prev_id:
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
        if index > 0 and (current_month != prev_month or current_year != prev_year):


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
    fm_dict = {
        'seqs': np.array(seqs),
        'cs': np.array(cs),
        'ts': np.array(ts),
        'cols': state_features,
        'rs': np.array(rs),
        'seqs_ts': np.array(seqs_ts)
    }
    with open(output_path, 'wb') as f:
        pickle.dump(fm_dict, f) 

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

main()