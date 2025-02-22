import math
import numpy as np
import pickle

N_SEQUENCES = 10000
DIM = 50 #20 for small, 50 for large
HORIZON = 100 #11 for small, 100 for large
mu = 0 #mean for initialization
b = -15 #-3 for small, -15 for large
w = math.sqrt(0.5)

output_path = '../data/LargeRW.pkl'

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def terminate(x, gamma):
    p = sigmoid(np.matmul(gamma, x) + b)
    sample = np.random.binomial(1, p, size=None)
    return sample

def random_walk(gamma):
    x = np.random.normal(loc=np.full((DIM,), mu), scale=np.ones((DIM,)), size=(DIM,)) #initialize state
    #walk until termination or reaching horizon (censored)
    seq = []
    r = []
    c = True
    t = 0
    for step in range(HORIZON):
        t += 1
        r.append(1)
        seq.append(x)
        if terminate(x, gamma):
            c = False
            break
        x = np.random.normal(loc=x, scale=np.full(x.shape, w), size=x.shape)
    seq_t = [i for i in range(len(seq), 0, -1)]
    return seq, c, t, r, seq_t

def main():
    seqs = []
    cs = []
    ts = []
    rs = []
    seqs_ts = []
    censor_count = 0
    seq_lens = []
    gamma = np.random.normal(loc=np.zeros((DIM,)), scale=np.ones((DIM,)), size=(DIM,))
    for seq_idx in range(N_SEQUENCES):
        seq, c, t, r, seq_t = random_walk(gamma)
        seqs.append(seq)
        seq_lens.append(len(seq))
        if c:
            censor_count += 1
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
    print(f"Censor Count (%): {censor_count} ({censor_count/N_SEQUENCES*100})")
    print(f"Median Length: {np.median(seq_lens)}")
    print(f"Mean Length (std.dev): {np.mean(seq_lens)} ({np.std(seq_lens)})")

main()
        
