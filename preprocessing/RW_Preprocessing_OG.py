import numpy as np
import pickle
from scipy.special import expit as sigmoid

#Random Walk generator from https://github.com/spotify-research/tdsurv/blob/main/notebooks/random-walk-final.ipynb

def make_generator(mat, sigma, sigma0, thetas, bias, rng):
    """Sample from a Gauss-Markov process.
    
    Initial state:
    
        x_0 = noise(sigma0)
    
    Transition dynamics:
    
        x_nxt = np.dot(mat, x) + noise(sigma)
        
    Churn probability:
    
        p = sigmoid(np.dot(x, thetas) + bias)
    """
    n_dims = mat.shape[0]
    def gen(n_samples, horizon):
        seqs = np.zeros((n_samples, horizon + 1, n_dims + 1))
        ts = np.zeros(n_samples, dtype=int)
        cs = np.zeros(n_samples, dtype=bool)
        rs = np.zeros((n_samples, horizon + 1), dtype=int)
        seqs_ts = np.zeros((n_samples, horizon + 1), dtype=int)
        for i in range(n_samples):
            seqs[i, 0] = np.append(sigma0 * rng.normal(size=n_dims), 1)
            ts[i] = 1
            for j in range(0, horizon):
                p = sigmoid(np.dot(seqs[i, j, :-1], thetas) + bias)
                if rng.uniform() < p:
                    break
                ts[i] += 1
                seqs[i, j + 1] = np.append(
                    np.dot(mat, seqs[i, j + 1, :-1]) + sigma * rng.normal(size=n_dims),
                    1,
                )
                rs[i, j] = 1
                seqs_ts[i,:j+1] += 1
            if ts[i] > horizon:
                ts[i] = horizon
                cs[i] = True
        return (seqs, ts, cs, rs, seqs_ts)
    return gen

output_path = '../data/LargeRW.pkl'

rng = np.random.default_rng(seed=0)
n_dims = 50 #20 for Small, 50 for Large
n_samples = 10000
horizon = 99 #10 for Small, 99 for Large

# Churn parameters.
thetas = rng.normal(size=n_dims)
bias = -8 #-3 for Small, -8 for Large

# Transition dynamics.
mat = 1.0 * np.eye(n_dims)
sigma = 0.5
sigma0 = 1.0 # Affect results quite a lot

gen = make_generator(mat, sigma, sigma0, thetas, bias, rng)
seqs, ts, cs, rs, seqs_ts = gen(n_samples=n_samples, horizon=horizon)

rw_dict = {
    'seqs': seqs,
    'cs': cs,
    'ts': ts,
    'cols': np.array([str(i) for i in range(n_dims+1)]),
    'rs': rs,
    'seqs_ts': seqs_ts
}
with open(output_path, 'wb') as f:
    pickle.dump(rw_dict, f) 

print(f"Censor Count (%): {cs.sum()} ({cs.sum()/n_samples*100})")
print(f"Median Length: {np.median(seqs_ts[:,0])}")
print(f"Mean Length (std.dev): {np.mean(seqs_ts[:,0])} ({np.std(seqs_ts[:,0])})")