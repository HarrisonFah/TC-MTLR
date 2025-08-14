import copy
import numpy as np
from dataclasses import dataclass
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from utils import MTLRDataGenerator, get_data, train_val_test_split, median_time_bins, quantile_time_bins
from SurvivalEVAL.Evaluator import SurvivalEvaluator

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# Implementation of TC-MTLR for sequences of states (no actions)
# Uses partial code from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/c51.py

@dataclass
class ConfigParams:
	"""A structure for configuration"""
	dataset_name: str
	batch_size: int
	learning_rate: float
	layer_size: int
	num_hidden: int
	use_quantiles: bool
	tau: float
	lambda_: int
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

class TC_MTLR(object):
	def __init__(
		self,
		config_kwargs,
		seed,
		discount=1.0,
		tau=0.1,
		policy_freq=1,
	):
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
		self.tau = self.config.tau
		self.lambda_ = self.config.lambda_

		self.discount = discount
		self.policy_freq = policy_freq
		self.batch_size = self.config.batch_size

		self.total_it = 0

	def init_networks(self, train_gen, val_gen, test_gen):
		if self.use_quantiles:
			time_bins = quantile_time_bins(train_gen, val_gen, test_gen, self.horizon)
		else:
			time_bins = median_time_bins(train_gen, val_gen, test_gen, self.horizon)
		self.time_bins = torch.tensor(time_bins).to(device).float()
		self.num_atoms = len(self.time_bins)
		self.MTLR_network = MTLR_network(self.state_dim, self.num_atoms, self.layer_size, self.num_hidden).to(device)
		self.MTLR_target = copy.deepcopy(self.MTLR_network)
		self.MTLR_optimizer = torch.optim.Adam(self.MTLR_network.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

	def calculate_isd_numerator(self, x, mask):
		lower_triangle = torch.tril(torch.full((x.shape[1], x.shape[1]), 1)).to(dtype=torch.float, device=device)
		sequences = torch.matmul(x, lower_triangle)
		masked_exp_sequences = torch.matmul(torch.exp(sequences), torch.transpose(mask, 0, 1))
		exp_diag = torch.diag(masked_exp_sequences, 0)
		return exp_diag

	def calculate_isd_denominator(self, x):
		lower_triangle = torch.tril(torch.full((x.shape[1], x.shape[1]), 1)).to(dtype=torch.float, device=device)
		sequences = torch.matmul(x, lower_triangle)
		exp_sum = torch.sum(torch.exp(sequences), dim=1)
		return exp_sum

	def get_sequence_probs(self, model, state):
		lower_triangle = torch.tril(torch.full((self.num_atoms, self.num_atoms), 1)).to(dtype=torch.float, device=device)
		preds = model(state)
		sequence_logits = torch.matmul(preds, lower_triangle)
		sequence_probs = F.softmax(sequence_logits, dim=1)
		return sequence_probs

	def train_step(self, state, next_state, reward, not_done, censors, times):
		self.total_it += 1

		current_probs = self.get_sequence_probs(self.MTLR_network, state)

		with torch.no_grad():
			# Compute the target Q value
			next_probs = self.get_sequence_probs(self.MTLR_target, next_state)

			z = self.time_bins
			bellman = (reward + self.discount * not_done * z) #calculates the bellman value for each time bin
			bellman = ((1-self.lambda_)*bellman + self.lambda_*(bellman*censors[:, None] + times.repeat((1, bellman.shape[1]))*(~censors[:, None]))).clamp(min(self.time_bins), max(self.time_bins)-(1e-6)) #calculates the bellman value for each time bin
			buckets = torch.bucketize(bellman, z, right=True) #gets the index of time bins that each bellman value falls into
			l = (buckets - 1).clip(min=0, max=(z.shape[0]-1)) #gets the lower index
			u = (l + 1).clip(min=1, max=(z.shape[0]-1)) #gets the upper index
			l_val = self.time_bins[l] #gets the value of lower time bin
			u_val = self.time_bins[u] #gets the value of upper time bin
			b = l + (bellman - l_val)/((u_val - l_val).clip(min=1e-5)) #finds the continuous 'index' of bellman value in the time bin array
			terminal_idxs = ~(not_done.bool()).repeat((1, z.shape[0]))
			b[terminal_idxs] = l[terminal_idxs].float() #set the index of termial transitions to the lower time bin
			b[l == u] = l[l == u].float() #set the index where the bellman value is greater than the horizon to the lower time bin

			#distributes probability to neighbors based on distance to each of them
			d_m_l = (u + (l == u).float() - b) * next_probs 
			d_m_u = (b - l) * next_probs

			m = torch.zeros((state.shape[0], self.num_atoms)).to(device)
			for i in range(state.shape[0]):
				m[i].index_add_(0, l[i].long(), d_m_l[i])
				m[i].index_add_(0, u[i].long(), d_m_u[i])

		q_loss = (-(m * current_probs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean() #cross entropy loss
		return_loss = q_loss.item()
		self.MTLR_optimizer.zero_grad()
		q_loss.backward()
		self.MTLR_optimizer.step()

		#update the target network
		if self.total_it % self.policy_freq == 0:
			for param, target_param in zip(self.MTLR_network.parameters(), self.MTLR_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		return return_loss

	def train(self, train_gen):
		losses = []
		for epoch in range(self.config.num_epochs):
			for batch in train_gen:
				state, next_state, reward, not_done, censor, time = batch
				state = torch.FloatTensor(state).to(device)
				next_state = torch.FloatTensor(next_state).to(device)
				reward = torch.FloatTensor(reward).to(device).reshape((-1,1))
				not_done = torch.FloatTensor(not_done).to(device).reshape((-1,1))
				censor = torch.BoolTensor(censor).to(device).reshape((-1))
				time = torch.FloatTensor(time).to(device).reshape((-1,1))

				loss = self.train_step(state, next_state, reward, not_done, censor, time)

				# log
				if epoch % self.config.log_interval == 0 and epoch > 1:
					print(f"Epoch: {epoch+1}/{self.config.num_epochs}")
					print(f"Train classification loss: {loss.item():.3f} at epoch {epoch}")
					print()

				losses.append(loss)
			train_gen.reset()

		return losses

	def get_isd(self, state):
		isd = torch.zeros((state.shape[0], self.num_atoms)).to(device=device)
		preds = self.MTLR_network(state)
		norm = self.calculate_isd_denominator(preds)
		mask = torch.ones((state.shape[0], self.num_atoms)).to(device=device)
		for i in range(self.num_atoms):
			logits = self.calculate_isd_numerator(preds, mask)
			isd[:,i] = logits/norm
			mask[:,i] = torch.zeros((state.shape[0],)).to(device=device)
		return isd

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

	def eval(self, train_gen, eval_gen, time_bins, lambda_cox=False):
		train_state, train_next_state, train_reward, train_not_done, train_times, train_censor = train_gen.get_initial_data()
		eval_state, eval_next_state, eval_reward, eval_not_done, eval_times, eval_censor = eval_gen.get_initial_data()
		eval_state = torch.tensor(eval_state).to(device)
		isds = self.get_isd(eval_state).detach().cpu().numpy()
		isds[:,-1] = np.zeros((isds.shape[0],))
		
		evaluator = SurvivalEvaluator(isds, self.time_bins, eval_times, ~eval_censor, train_times, ~train_censor)
		predicted_times = evaluator.predict_time_from_curve(evaluator.predict_time_method)


		cindex, concordant_pairs, total_pairs = evaluator.concordance(ties="None")
		ibs = evaluator.integrated_brier_score(num_points=isds.shape[1], IPCW_weighted=True, draw_figure=False)
		mae_uncensored = evaluator.mae(method='Uncensored')
		mae_hinge = evaluator.mae(method='Hinge')
		maepo = evaluator.mae(method='Pseudo_obs', weighted=True, truncated_time=np.max(eval_times))

		return isds, cindex, ibs, mae_uncensored, mae_hinge, maepo

		