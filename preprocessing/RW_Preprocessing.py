import math
import numpy as np
import pickle

N_SEQUENCES = 10000
DIM = 20
HORIZON = 10 #10 for small, 100 for large
b = -3
w = 0.5

output_path = '../data/SmallRW.pkl'

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def terminate(x):
    gamma = np.random.normal(size=(DIM,))
    prob = sigmoid(np.matmul(gamma, x) + b)
    return prob > 0.5

def random_walk():
    x = np.random.normal(loc=np.zeros((DIM,)), scale=np.ones((DIM,)), size=(DIM,)) #initialize state
    #walk until termination or reaching horizon (censored)
    seq = []
    r = []
    c = True
    t = 0
    for step in range(HORIZON):
        t += 1
        r.append(1)
        seq.append(x)
        if terminate(x):
            c = False
            break
        x = np.random.normal(loc=x, scale=np.full(x.shape, w**2), size=x.shape)
    seq_t = [i for i in range(len(seq), 0, -1)]
    return seq, c, t, r, seq_t

def main():
    seqs = []
    cs = []
    ts = []
    rs = []
    seqs_ts = []
    for seq_idx in range(N_SEQUENCES):
        seq, c, t, r, seq_t = random_walk()
        seqs.append(seq)
        cs.append(c)
        ts.append(t)
        rs.append(r)
        seqs_ts.append(seq_t)
    #pads arrays with zeros so they all have same length
    for seq_idx in range(len(seqs)):
        missing_states = HORIZON-len(seqs[seq_idx])
        rs[seq_idx] += [0 for state in range(missing_states)]
        seqs_ts[seq_idx] += [0 for state in range(missing_states)]
        seqs[seq_idx] = seqs[seq_idx] + [[0 for feature in range(DIM)] for state in range(missing_states)]
    rw_dict = {
        'seqs': np.array(seqs),
        'cs': np.array(cs),
        'ts': np.array(ts),
        'cols': np.array([str(i) for i in range(DIM)]),
        'rs': np.array(rs),
        'seqs_ts': np.array(seqs_ts)
    }
    with open(output_path, 'wb') as f:
        pickle.dump(rw_dict, f) 

main()
        
