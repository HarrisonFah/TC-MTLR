import numpy as np
from scipy.interpolate import interp1d
import torch
import torch.nn as nn

from math import ceil, sqrt
from typing import Optional, Union

from math import sqrt, ceil
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, roc_auc_score
from tqdm import tqdm, trange

from dataclasses import dataclass
import inspect
from utils import MTLRDataGenerator, get_data, train_val_test_split, median_time_bins, quantile_time_bins
from SurvivalEVAL.Evaluator import SurvivalEvaluator

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

@dataclass
class ConfigParams:
	"""A structure for configuration"""
	dataset_name: str
	batch_size: int
	learning_rate: float
	C1: float
	layer_size: int
	num_hidden: int
	use_quantiles: bool
	log_interval: int
	weight_decay: float
	num_epochs: int
	dataset_kwargs: dict
	axis: int
	arch: dict
	preprocessed_data: bool
	verbose: bool
	calculate_tgt_and_mask: bool = True
	landmark: bool = False
	output_file: str = None
	ckpt_path: str = None
	# horizon: int

	@classmethod
	def from_dict(cls, env):
		"""To ignore args that are not in the class,
		see XXXX
		"""
		return cls(**{
			k: v for k, v in env.items()
			if k in inspect.signature(cls).parameters
		})

class MTLR_network(nn.Module):
	def __init__(self, state_dim, num_outputs, layer_size=128, num_hidden=1):
		super(MTLR_network, self).__init__()

		self.l0 = nn.Linear(state_dim, num_outputs)
		self.l1 = nn.Linear(state_dim, layer_size)
		self.l2 = nn.Linear(layer_size, layer_size)
		self.l3 = nn.Linear(layer_size, layer_size)
		self.l4 = nn.Linear(layer_size, num_outputs)

		self.num_hidden = num_hidden

	def forward(self, state):
		if self.num_hidden == 0:
			q = self.l0(state)
		else:
			q = F.elu(self.l1(state))
			if self.num_hidden > 1:
				q = F.elu(self.l2(q))
			if self.num_hidden > 2:
				q = F.elu(self.l3(q))
			q = self.l4(q)
		return q

class MTLR(nn.Module):
	"""Multi-task logistic regression for individualised
	survival prediction.

	The MTLR time-logits are computed as:
	`z = sum_k x^T w_k + b_k`,
	where `w_k` and `b_k` are learnable weights and biases for each time
	interval.

	Note that a slightly more efficient reformulation is used here, first
	proposed in [2]_.

	References
	----------
	..[1] C.-N. Yu et al., ‘Learning patient-specific cancer survival
	distributions as a sequence of dependent regressors’, in Advances in neural
	information processing systems 24, 2011, pp. 1845–1853.
	..[2] P. Jin, ‘Using Survival Prediction Techniques to Learn
	Consumer-Specific Reservation Price Distributions’, Master's thesis,
	University of Alberta, Edmonton, AB, 2015.
	"""

	def __init__(self, config_kwargs, seed):
		super().__init__()

		self.config = ConfigParams.from_dict(config_kwargs)
		self.seed = seed
		H = self.config.dataset_kwargs['horizon']
		self.horizon = H
		self.calculate_tgt_and_mask_at_epoch = not self.config.calculate_tgt_and_mask

		seqs, ts, cs, h_tgt, h_ws, mask, rs, seqs_ts = get_data(self.config.dataset_name,
														self.config.landmark,
														self.config.calculate_tgt_and_mask,
														self.config.dataset_kwargs)
		seqs = seqs.astype(np.float32)

		self.data = {'seqs': seqs,
					 'ts': ts,
					 'cs': cs,
					 'h_ws': h_ws,
					 'target': h_tgt,
					 'mask': mask,
					 'rs': rs,
					 'seqs_ts': seqs_ts}

		self.state_dim = seqs.shape[-1]
		self.layer_size = self.config.layer_size
		self.num_hidden = self.config.num_hidden
		self.use_quantiles = self.config.use_quantiles

		self.column_names = []
		for i in range(self.state_dim):
			self.column_names.append('feature_' + str(i))
		self.column_names += ['time', 'event']

	def init_networks(self, train_gen):
		if self.use_quantiles:
			time_bins = quantile_time_bins(train_gen, self.horizon)
		else:
			time_bins = median_time_bins(train_gen, self.horizon)
		self.time_bins = torch.tensor(time_bins, dtype=torch.float).to(device)
		self.num_time_bins = len(self.time_bins) + 1
		self.MTLR_network = MTLR_network(self.state_dim, self.num_time_bins, self.layer_size, self.num_hidden).to(device)
		self.MTLR_optimizer = torch.optim.Adam(self.MTLR_network.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

	def get_train_val_test(self, val_size=.15, test_size=.2, num_train_seqs=None):
		data_manager = MTLRDataGenerator

		X_train, X_val, X_test, y_train, y_val, y_test, hws_train, hws_val, hws_test, \
		m_train, m_val, m_test, ts_train, ts_val, ts_test, cs_train, cs_val, cs_test, \
		rs_train, rs_val, rs_test, seqs_ts_train, seqs_ts_val, seqs_ts_test = train_val_test_split(self.data['seqs'],
																	self.data['target'],
																	self.data['h_ws'],
																	self.data['mask'],
																	self.data['ts'],
																	self.data['cs'],
																	self.data['rs'],
																	self.data['seqs_ts'],
																	seed=self.seed,
																	val_size=val_size,
																	test_size=test_size,
																	num_train_seqs=num_train_seqs)

		train_gen = data_manager(X=X_train,
								 ts=ts_train, cs=cs_train,
								 y=y_train, rs=rs_train,
								 seqs_ts=seqs_ts_train, mask=m_train,
								 batch_size=self.config.batch_size)
		
		val_gen = data_manager(X=X_val,
								 ts=ts_val, cs=cs_val,
								 y=y_val, rs=rs_val,
								 seqs_ts=seqs_ts_val, mask=m_val,
								 batch_size=self.config.batch_size)
		
		test_gen = data_manager(X=X_test,
								ts=ts_test, cs=cs_test,
								y=y_test, rs=rs_test,
								seqs_ts=seqs_ts_test, mask=m_test,
								batch_size=self.config.batch_size)

		return train_gen, val_gen, test_gen

		# training functions
	def train(self, train_gen, verbose=False):
		"""Trains the MTLR model using minibatch gradient descent.
		
		Parameters
		----------
		model : torch.nn.Module
			MTLR model to train.
		data_train : pd.DataFrame
			The training dataset. Must contain a `time` column with the
			event time for each sample and an `event` column containing
			the event indicator.
		num_epochs : int
			Number of training epochs.
		lr : float
			The learning rate.
		weight_decay : float
			Weight decay strength for all parameters *except* the MTLR
			weights. Only used for Deep MTLR training.
		C1 : float
			L2 regularization (weight decay) strenght for MTLR parameters.
		batch_size : int
			The batch size.
		verbose : bool
			Whether to display training progress.
		device : str
			Device name or ID to use for training.
			
		Returns
		-------
		torch.nn.Module
			The trained model.
		"""
		state, next_state, reward, not_done, time, censor = train_gen.get_all_data()
		state = torch.FloatTensor(state).to(device)
		time = torch.FloatTensor(time).to(device).reshape((-1,1))
		censor = torch.BoolTensor(censor).to(device).reshape((-1,1))

		data_train_tensor = torch.cat((state, time.reshape(-1, 1), (~censor).reshape(-1, 1)), dim=1)
		data_train = pd.DataFrame(data_train_tensor.cpu(), columns=self.column_names)

		x = torch.tensor(data_train.drop(["time", "event"], axis=1).values, dtype=torch.float)
		y = encode_survival(data_train["time"].values, data_train["event"].values, self.time_bins.cpu())
		train_loader = DataLoader(TensorDataset(x, y), batch_size=self.config.batch_size, shuffle=True)
		
		pbar =  trange(self.config.num_epochs, disable=not verbose)
		for i in pbar:
			for xi, yi in train_loader:
				xi, yi = xi.to(device), yi.to(device)
				y_pred = self.MTLR_network(xi)
				loss = mtlr_neg_log_likelihood(y_pred, yi, self, C1=self.config.C1, average=True)
				self.MTLR_optimizer.zero_grad()
				loss.backward()
				self.MTLR_optimizer.step()
			pbar.set_description(f"[epoch {i+1: 4}/{self.config.num_epochs}]")
			pbar.set_postfix_str(f"loss = {loss.item():.4f}")

	def eval(self, train_gen, eval_gen, time_bins, lambda_cox=False):
		state, next_state, reward, not_done, time, censor = train_gen.get_all_data()
		state = torch.FloatTensor(state).to(device)
		time = torch.FloatTensor(time).to(device).reshape((-1,1))
		censor = torch.BoolTensor(censor).to(device).reshape((-1,1))
		data_train_tensor = torch.cat((state, time.reshape(-1, 1), (~censor).reshape(-1, 1)), dim=1)
		data_train = pd.DataFrame(data_train_tensor.cpu(), columns=self.column_names)

		state, next_state, reward, not_done, time, censor = eval_gen.get_all_data()
		state = torch.FloatTensor(state).to(device)
		time = torch.FloatTensor(time).to(device).reshape((-1,1))
		censor = torch.BoolTensor(censor).to(device).reshape((-1,1))
		data_test_tensor = torch.cat((state, time.reshape(-1, 1), (~censor).reshape(-1, 1)), dim=1)
		data_test = pd.DataFrame(data_test_tensor.cpu(), columns=self.column_names)

		x = torch.tensor(data_test.drop(["time", "event"], axis=1).values, dtype=torch.float, device=device)
		logits = self.MTLR_network(x)
		isds = mtlr_survival(logits).detach().cpu().numpy()
		isds[:,isds.shape[1]-1] = np.zeros((isds.shape[0]))
		time_bins = time_bins.detach().cpu().numpy()
		time_bins = np.concatenate((time_bins, np.array([time_bins[len(time_bins)-1]+1])))
		evaluator = SurvivalEvaluator(isds, time_bins, data_test["time"], data_test["event"], data_train["time"], data_train["event"])

		cindex, concordant_pairs, total_pairs = evaluator.concordance(ties="None")
		ibs = evaluator.integrated_brier_score(num_points=isds.shape[1], IPCW_weighted=True, draw_figure=False)
		mae_uncensored = evaluator.mae(method='Uncensored')
		mae_hinge = evaluator.mae(method='Hinge')
		mae_po = evaluator.mae(method='Pseudo_obs', weighted=True)

		return isds, cindex, ibs, mae_uncensored, mae_hinge, mae_po


def masked_logsumexp(x: torch.Tensor,
					 mask: torch.Tensor,
					 dim: int = -1) -> torch.Tensor:
	"""Computes logsumexp over elements of a tensor specified by a mask
	in a numerically stable way.

	Parameters
	----------
	x
		The input tensor.
	mask
		A tensor with the same shape as `x` with 1s in positions that should
		be used for logsumexp computation and 0s everywhere else.
	dim
		The dimension of `x` over which logsumexp is computed. Default -1 uses
		the last dimension.

	Returns
	-------
	torch.Tensor
		Tensor containing the logsumexp of each row of `x` over `dim`.
	"""
	max_val, _ = (x * mask).max(dim=dim)
	max_val = torch.clamp_min(max_val, 0)
	return torch.log(
		torch.sum(torch.exp(x - max_val.unsqueeze(dim)) * mask,
				  dim=dim)) + max_val


def mtlr_neg_log_likelihood(logits: torch.Tensor,
							target: torch.Tensor,
							model: torch.nn.Module,
							C1: float,
							average: bool = False) -> torch.Tensor:
	"""Computes the negative log-likelihood of a batch of model predictions.

	Parameters
	----------
	logits : torch.Tensor, shape (num_samples, num_time_bins)
		Tensor with the time-logits (as returned by the MTLR module) for one
		instance in each row.
	target : torch.Tensor, shape (num_samples, num_time_bins)
		Tensor with the encoded ground truth survival.
	model
		PyTorch Module with at least `MTLR` layer.
	C1
		The L2 regularization strength.
	average
		Whether to compute the average log likelihood instead of sum
		(useful for minibatch training).

	Returns
	-------
	torch.Tensor
		The negative log likelihood.
	"""
	censored = target.sum(dim=1) > 1
	nll_censored = masked_logsumexp(logits[censored], target[censored]).sum() if censored.any() else 0
	nll_uncensored = (logits[~censored] * target[~censored]).sum() if (~censored).any() else 0

	# the normalising constant
	norm = torch.logsumexp(logits, dim=1).sum()

	nll_total = -(nll_censored + nll_uncensored - norm)
	if average:
		nll_total = nll_total / target.size(0)

	# L2 regularization
	for k, v in model.named_parameters():
		if "mtlr_weight" in k:
			nll_total += C1/2 * torch.sum(v**2)

	return nll_total


def mtlr_survival(logits: torch.Tensor) -> torch.Tensor:
	"""Generates predicted survival curves from predicted logits.

	Parameters
	----------
	logits
		Tensor with the time-logits (as returned by the MTLR module) for one
		instance in each row.

	Returns
	-------
	torch.Tensor
		The predicted survival curves for each row in `pred` at timepoints used
		during training.
	"""
	# TODO: do not reallocate G in every call
	G = torch.tril(torch.ones(logits.size(1),
							  logits.size(1))).to(logits.device)
	density = torch.softmax(logits, dim=1)
	return torch.matmul(density, G)


def mtlr_survival_at_times(logits: torch.Tensor,
						   train_times: Union[torch.Tensor, np.ndarray],
						   pred_times: np.ndarray) -> np.ndarray:
	"""Generates predicted survival curves at arbitrary timepoints using linear
	interpolation.

	Notes
	-----
	This function uses scipy.interpolate internally and returns a Numpy array,
	in contrast with `mtlr_survival`.

	Parameters
	----------
	logits
		Tensor with the time-logits (as returned by the MTLR module) for one
		instance in each row.
	train_times
		Time bins used for model training. Must have the same length as the
		first dimension of `pred`.
	pred_times
		Array of times used to compute the survival curve.

	Returns
	-------
	np.ndarray
		The survival curve for each row in `pred` at `pred_times`. The values
		are linearly interpolated at timepoints not used for training.
	"""
	train_times = np.pad(train_times, (1, 0))
	surv = mtlr_survival(logits).detach().cpu().numpy()
	interpolator = interp1d(train_times, surv)
	return interpolator(np.clip(pred_times, 0, train_times.max()))


def mtlr_hazard(logits: torch.Tensor) -> torch.Tensor:
	"""Computes the hazard function from MTLR predictions.

	The hazard function is the instantenous rate of failure, i.e. roughly
	the risk of event at each time interval. It's computed using
	`h(t) = f(t) / S(t)`,
	where `f(t)` and `S(t)` are the density and survival functions at t,
	respectively.

	Parameters
	----------
	logits
		The predicted logits as returned by the `MTLR` module.

	Returns
	-------
	torch.Tensor
		The hazard function at each time interval in `y_pred`.
	"""
	return torch.softmax(
		logits, dim=1)[:, :-1] / (mtlr_survival(logits) + 1e-15)[:, 1:]


def mtlr_risk(logits: torch.Tensor) -> torch.Tensor:
	"""Computes the overall risk of event from MTLR predictions.

	The risk is computed as the time integral of the cumulative hazard,
	as defined in [1]_.

	Parameters
	----------
	logits
		The predicted logits as returned by the `MTLR` module.

	Returns
	-------
	torch.Tensor
		The predicted overall risk.
	"""
	hazard = mtlr_hazard(logits)
	return torch.sum(hazard.cumsum(1), dim=1)

from math import ceil, sqrt
from typing import Optional, Union

import numpy as np
import torch


TensorOrArray = Union[torch.Tensor, np.ndarray]


def encode_survival(time: Union[float, int, TensorOrArray],
					event: Union[int, bool, TensorOrArray],
					bins: TensorOrArray) -> torch.Tensor:
	"""Encodes survival time and event indicator in the format
	required for MTLR training.

	For uncensored instances, one-hot encoding of binned survival time
	is generated. Censoring is handled differently, with all possible
	values for event time encoded as 1s. For example, if 5 time bins are used,
	an instance experiencing event in bin 3 is encoded as [0, 0, 0, 1, 0], and
	instance censored in bin 2 as [0, 0, 1, 1, 1]. Note that an additional
	'catch-all' bin is added, spanning the range `(bins.max(), inf)`.

	Parameters
	----------
	time
		Time of event or censoring.
	event
		Event indicator (0 = censored).
	bins
		Bins used for time axis discretisation.

	Returns
	-------
	torch.Tensor
		Encoded survival times.
	"""
	# TODO this should handle arrays and (CUDA) tensors
	if isinstance(time, (float, int, np.ndarray)):
		time = np.atleast_1d(time)
		time = torch.tensor(time)
	if isinstance(event, (int, bool, np.ndarray)):
		event = np.atleast_1d(event)
		event = torch.tensor(event)

	if isinstance(bins, np.ndarray):
		bins = torch.tensor(bins)

	try:
		device = bins.device
	except AttributeError:
		device = "cpu"

	time = np.clip(time, 0, bins.max())
	# add extra bin [max_time, inf) at the end
	y = torch.zeros((time.shape[0], bins.shape[0] + 1),
					dtype=torch.float,
					device=device)
	# For some reason, the `right` arg in torch.bucketize
	# works in the _opposite_ way as it does in numpy,
	# so we need to set it to True
	bin_idxs = torch.bucketize(time, bins, right=True)
	for i, (bin_idx, e) in enumerate(zip(bin_idxs, event)):
		if e == 1:
			y[i, bin_idx] = 1
		else:
			y[i, bin_idx:] = 1
	return y.squeeze()
