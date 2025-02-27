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
sklearn.set_config(transform_output="default")
import category_encoders as ce
import datetime

input_path = '../data/lastfm-dataset-1k/userid-timestamp-artid-artname-traid-traname.tsv'
output_path = '../data/LastFM.pkl'

columns = ['userid', 'timestamp', 'musicbrainz-artist-id', 'artist-name', 'musicbrainz-track-id', 'track-name']

state_features = ['top_song', 'top_artist', 'num_songs', 'num_unique']

def diff_month(d1, d2):
    return abs((d1.year - d2.year) * 12 + d1.month - d2.month)

#returns the most frequent song/artist, the total number of songs listened to, and the number of unique songs
def get_features(song_counts):
    sorted_counts = sorted(song_counts, key=song_counts.get, reverse=True)
    top_song, top_artist = sorted_counts[0]
    num_songs = sum(song_counts.values())
    num_unique = len(song_counts.keys())
    return (top_song, top_artist, num_songs, num_unique)

def main():
    df = pd.read_csv(input_path, sep='\t', header=None, on_bad_lines='skip')
    df.columns = columns
    print(df.describe())

    n_unique_artist = df['musicbrainz-artist-id'].nunique()
    artist_encoder = ce.BaseNEncoder(base=n_unique_artist)
    encoded_artists = artist_encoder.fit_transform(df['musicbrainz-artist-id'])["musicbrainz-artist-id_1"].to_list()

    n_unique_song = df['musicbrainz-track-id'].nunique()
    song_encoder = ce.BaseNEncoder(base=n_unique_song)
    encoded_songs = song_encoder.fit_transform(df['musicbrainz-track-id'])["musicbrainz-track-id_1"].to_list()

    all_rs = [] #used to get statistics of all consecutive time between states

    seqs = []
    cs = []
    ts = []
    rs = []
    seqs_ts = []
    horizon = -1

    seq = []
    c = None
    r = []
    seq_t = []

    prev_id = -1
    prev_month = 0
    prev_year = 0
    end_date = None

    song_counts = {} #stores pairs of ((artist, song): count)

    #Iterate through all rows to build sequence data
    #Note that date is in reverse order (starts with most recent date)
    #So we have to reverse the sequences and update the time-to-events
    for index, row in df.iterrows():
        current_id = row['userid']
        current_date = datetime.datetime.fromisoformat(row['timestamp'].rstrip('Z'))
        current_month = current_date.month
        current_year = current_date.year
        #if reached the next user, save the previous sequence
        if index > 0 and current_id != prev_id:
            print(f"Index:{index}, ID:{current_id}")
            if len(seq) > 1:
                seq.reverse()
                seqs.append(seq)
                cs.append(c)
                ts.append(len(seq))
                r.reverse()
                rs.append(r)
                for seq_idx in range(len(seq_t)):
                    seq_t[seq_idx] = diff_month(prev_date, end_date) - seq_t[seq_idx]
                seqs_ts.append(seq_t)
                if len(seq) > horizon:
                    horizon = len(seq)

            seq = []
            c = None
            r = []
            seq_t = []
            prev_day = 0
            song_counts = {}
            end_date = None

        #if reached the next month, calculate features and times
        if index > 0 and end_date is not None and (current_month != prev_month or current_year != prev_year):
            state = get_features(song_counts)
            seq.append(state)
            r.append(diff_month(current_date, prev_date))
            all_rs.append(diff_month(current_date, prev_date))
            seq_t.append(diff_month(current_date, end_date))
            song_counts = {}
        
        artist_code = encoded_artists[index] #artist_encoder.transform(row['musicbrainz-artist-id'])
        song_code = encoded_songs[index] #song_encoder.transform(row['musicbrainz-track-id'])
        try:
            song_counts[(artist_code, song_code)] += 1
        except:
            song_counts[(artist_code, song_code)] = 1
        #if listened to songs in june 2009 then censored
        if c is None:
            if current_year == 2009 and current_month == 6:
                c = True
            else:
                c = False
        if end_date is None:
            end_date = current_date
        prev_id = current_id
        prev_date = current_date
        prev_month = current_month
        prev_year = current_year
    #add data from final sequence
    if len(seq) > 1:
        seq.reverse()
        seqs.append(seq)
        cs.append(c)
        ts.append(len(seq))
        r.reverse()
        rs.append(r)
        seqs_ts.append(seq_t)
        if len(seq) > horizon:
            horizon = len(seq)

    print('Horizon:', horizon)

    #pads arrays with zeros so they all have same length
    for seq_idx in range(len(seqs)):
        missing_states = horizon-len(seqs[seq_idx])
        rs[seq_idx] += [0 for state in range(missing_states)]
        seqs_ts[seq_idx] += [0 for state in range(missing_states)]
        seqs[seq_idx] = seqs[seq_idx] + [[0 for feature in range(len(state_features))] for state in range(missing_states)]
    #normalizes features
    seqs = np.array(seqs)
    means = np.mean(seqs.reshape((-1,seqs.shape[2])), axis=0)
    stds = np.std(seqs.reshape((-1,seqs.shape[2])), axis=0)
    seqs = (seqs - means) / stds
    fm_dict = {
        'seqs': seqs,
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

    print("Statistics:")
    print("\tDataset Size:", np.array(seqs).shape[0])
    print("\tHorizon:", np.array(seqs).shape[1])
    print("\t# Features:", np.array(seqs).shape[2])
    print("\tMedian Length:", np.median(ts))
    print("\tMean Length:", np.mean(ts))
    print("\tStd. Dev. Length:", np.std(ts))
    print("\t% Censored:", np.mean(cs)*100)

main()