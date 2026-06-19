import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from SurvivalEVAL.Evaluator import SurvivalEvaluator

"""
The neural network used as the core of the TC-MTLR algorithm
"""
class NeuralNetwork(nn.Module):
	def __init__(self, state_dim, num_outputs, layer_size=128, num_hidden=1):
		super(NeuralNetwork, self).__init__()

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
	

"""
TC-MTLR class that includes both training and inference methods
"""
class TC_MTLR(object):
	"""
	Parameters:
		dataset: 
			The training dataset. 
			This is used to calculate the neural network input size and the quantile time bins.
			Should be a tuple of matrices (state, reward, next_state, censor, not_done, time_to_event)
				state (num_samples, num_features) 
				reward (num_samples, 1): is the time between the state and next state
				next_state (num_samples, num_features) 
				censor (num_samples, 1): 1 if censored, 0 otherwise
				not_done (num_samples, 1): 1 if not terminal state, 0 otherwise
				time_to_event (num_samples, 1): time from state to event
		layer_size: The size of the hidden layers of the neural network
		num_hidden: The number of hidden layers in the neural network
		use_quantiles: Whether to calculate time bins as uniform or quantiles.
		num_time_bins: Number of time bins
		discount: Discount factor used for calculating the Bellman error. Generally should be left as 1.
		tau: Hyperparameter used to determine how much the target network is updated from the main network.
		polcy_freq: How often to update the target network
		lambda_: Hyperparemter used to weight between bootstrapped targets and ground truth times (0 = only bootstrap, 1 = only ground truth)
		learning_rate: Learning rate used in training
		weight_decay: Weight decay factor used in Adam optimizer
	"""
	def __init__(
		self,
		dataset,
		layer_size=128,
		num_hidden=1,
		use_quantiles=True,
		num_time_bins=25,
		discount=1.0,
		tau=0.1,
		policy_freq=1,
		lambda_=0,
		learning_rate=0.01,
		weight_decay=0.4
	):
		self.state_dim = dataset[0].shape[-1]
		self.layer_size = layer_size
		self.num_hidden = num_hidden
		self.tau = tau
		self.lambda_ = lambda_

		self.discount = discount
		self.policy_freq = policy_freq
		self.num_time_bins = num_time_bins
		
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
		if use_quantiles:
			time_bins = self.quantile_time_bins(dataset, self.num_time_bins-1)
		else:
			time_bins = self.median_time_bins(dataset, self.num_time_bins-1)
		self.time_bins = torch.tensor(time_bins).to(self.device).float()
		self.neural_network = NeuralNetwork(self.state_dim, self.num_time_bins, self.layer_size, self.num_hidden).to(self.device)
		self.neural_network_target = copy.deepcopy(self.neural_network)
		self.optimizer = torch.optim.Adam(self.neural_network.parameters(), lr=learning_rate, weight_decay=weight_decay)

		self.total_it = 0
		
	"""
	Calculates the value for each time bin using quantiles of the time_to_events of the training dataset
	"""
	def quantile_time_bins(self, dataset, num_time_bins):
		times = dataset[5]
		uniform = np.random.uniform(-1e-2, 1e-2, size=times.shape)
		times = times.astype(uniform.dtype) + np.random.uniform(-1e-2, 1e-2, size=times.shape) #modifies times slightly so that the quantile bins do not have duplicates
		bins = np.quantile(times, np.linspace(0, 1, num_time_bins+1))
		bins[0] = 0
		return bins
	
	"""
	Calculates the value for each time bin using uniform spacing between each time bin up to the maximum observed time to event
	"""
	def median_time_bins(self, dataset, num_time_bins):
		times = dataset[5]
		max = np.max(times)
		time_bins = []
		for idx in range(num_time_bins+1):
			time_bins.append(max*idx/num_time_bins)
		return time_bins

	"""
	Calculates the numerator term for the Individualized Survival Distribution (ISD), seen in equation 2 of the TC-MTLR paper
	"""
	def calculate_isd_numerator(self, x, mask):
		lower_triangle = torch.tril(torch.full((x.shape[1], x.shape[1]), 1)).to(dtype=torch.float, device=self.device)
		sequences = torch.matmul(x, lower_triangle)
		masked_exp_sequences = torch.matmul(torch.exp(sequences), torch.transpose(mask, 0, 1))
		exp_diag = torch.diag(masked_exp_sequences, 0)
		return exp_diag

	"""
	Calculates the denominator term for the Individualized Survival Distribution (ISD), seen in equation 2 of the TC-MTLR paper
	"""
	def calculate_isd_denominator(self, x):
		lower_triangle = torch.tril(torch.full((x.shape[1], x.shape[1]), 1)).to(dtype=torch.float, device=self.device)
		sequences = torch.matmul(x, lower_triangle)
		exp_sum = torch.sum(torch.exp(sequences), dim=1)
		return exp_sum
	
	"""
	Calculates the probability distribution function from the neural network outputs, seen in equation 1 of the TC-MTLR paper
	"""
	def get_sequence_probs(self, model, state):
		lower_triangle = torch.tril(torch.full((self.num_time_bins, self.num_time_bins), 1)).to(dtype=torch.float, device=self.device)
		preds = model(state)
		sequence_logits = torch.matmul(preds, lower_triangle)
		sequence_probs = F.softmax(sequence_logits, dim=1)
		return sequence_probs

	"""
	Performs a single training step of the algorithm given a batch of training samples
	"""
	def train_step(self, state, next_state, reward, not_done, censors, times):
		self.total_it += 1

		current_probs = self.get_sequence_probs(self.neural_network, state)

		with torch.no_grad():
			# Compute the target Q value
			next_probs = self.get_sequence_probs(self.neural_network_target, next_state)

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

			m = torch.zeros((state.shape[0], self.num_time_bins)).to(device=self.device)
			for i in range(state.shape[0]):
				m[i].index_add_(0, l[i].long(), d_m_l[i])
				m[i].index_add_(0, u[i].long(), d_m_u[i])

		q_loss = (-(m * current_probs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean() #cross entropy loss
		return_loss = q_loss.item()
		self.optimizer.zero_grad()
		q_loss.backward()
		self.optimizer.step()

		#update the target network
		if self.total_it % self.policy_freq == 0:
			for param, target_param in zip(self.neural_network.parameters(), self.neural_network_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		return return_loss

	"""
	Trains the TC-MTLR model on the training dataset over multiple epochs and batches
	"""
	def train(self, dataset, batch_size=25, num_epochs=100):
		losses = []
		idxs = np.arange(dataset[0].shape[0])
		for epoch in range(num_epochs):
			np.random.shuffle(idxs)
			batch_idxs = idxs.reshape((-1, batch_size))
			for idx in batch_idxs:
				state = dataset[0][idx]
				reward = dataset[1][idx]
				next_state = dataset[2][idx]
				censor = dataset[3][idx]
				not_done = dataset[4][idx]
				time = dataset[5][idx]
				
				state = torch.FloatTensor(state).to(device=self.device)
				next_state = torch.FloatTensor(next_state).to(device=self.device)
				reward = torch.FloatTensor(reward).to(device=self.device).reshape((-1,1))
				not_done = torch.FloatTensor(not_done).to(device=self.device).reshape((-1,1))
				censor = torch.BoolTensor(censor).to(device=self.device).reshape((-1))
				time = torch.FloatTensor(time).to(device=self.device).reshape((-1,1))

				loss = self.train_step(state, next_state, reward, not_done, censor, time)
			losses.append(loss)
			print(f"Epoch: {epoch+1}/{num_epochs}")
			print(f"Train classification loss: {loss:.3f} at epoch {epoch}")
			print()

		return losses

	"""
	Calculates the Individualized Survival Distribution (ISD), seen in equation 2 of the TC-MTLR paper
	"""
	def get_isd(self, state):
		isd = torch.zeros((state.shape[0], self.num_time_bins)).to(device=self.device)
		preds = self.neural_network(state)
		norm = self.calculate_isd_denominator(preds)
		mask = torch.ones((state.shape[0], self.num_time_bins)).to(device=self.device)
		for i in range(self.num_time_bins):
			logits = self.calculate_isd_numerator(preds, mask)
			isd[:,i] = logits/norm
			mask[:,i] = torch.zeros((state.shape[0],)).to(device=self.device)
		return isd

	"""
	Calculates various survival evaluation metrics given a training/validation/test set
	"""
	def eval(self, train_dataset, test_dataset):
		train_censor = train_dataset[3]
		train_time = train_dataset[5]

		test_state = test_dataset[0]
		test_censor = test_dataset[3]
		test_time = test_dataset[5]

		test_state = torch.FloatTensor(test_state).to(device=self.device)
		isds = self.get_isd(test_state).detach().cpu().numpy()
		isds[:, 0] = np.ones((isds.shape[0],)) #ensures that the first probability is 1
		isds[:,-1] = np.zeros((isds.shape[0],)) #ensures that the last probability is 0
		
		evaluator = SurvivalEvaluator(isds, self.time_bins, test_time, ~test_censor, train_time, ~train_censor)
		predicted_times = evaluator.predict_time_from_curve(evaluator.predict_time_method)

		cindex, concordant_pairs, total_pairs = evaluator.concordance(ties="None")
		ibs = evaluator.integrated_brier_score(num_points=isds.shape[1], IPCW_weighted=True, draw_figure=False)
		mae_uncensored = evaluator.mae(method='Uncensored')
		mae_hinge = evaluator.mae(method='Hinge')
		maepo = evaluator.mae(method='Pseudo_obs', weighted=True, truncated_time=np.max(test_time))

		return isds, cindex, ibs, mae_uncensored, mae_hinge, maepo
	
"""
An example of how to use the above code to train a model and use it to generate predictions
"""
if __name__ == "__main__":
	#dataset of format (state, reward, next_state, censor, not_done, time_to_event)
	train_dataset = (
		# state 
		np.array([
			[8, 8, 8, 8],
			[6, 6, 6, 6],
			[3, 3, 3, 3],
			[1, 1, 1, 1]
		]),
		# reward
		np.array([
			2,
			3,
			2,
			1
		]),
		# next_state
		np.array([
			[6, 6, 6, 6],
			[3, 3, 3, 3],
			[1, 1, 1, 1],
			[0, 0, 0, 0]
		]),
		# censor
		np.array([
			0,
			0,
			0,
			0
		]),
		# not_done
		np.array([
			1,
			1,
			1,
			0
		]),
		# time_to_event
		np.array([
			8,
			6,
			3,
			1
		])
	)

	test_dataset = (
		# state 
		np.array([
			[9, 9, 9, 9],
			[5, 5, 5, 5],
		]),
		# reward (doesn't actually matter for test data)
		np.array([
			4,
			5,
		]),
		# next_state (doesn't actually matter for test data)
		np.array([
			[5, 5, 5, 5],
			[0, 0, 0, 0]
		]),
		# censor
		np.array([
			0,
			0,
		]),
		# not_done (doesn't actually matter for test data)
		np.array([
			1,
			0
		]),
		# time_to_event
		np.array([
			9,
			5,
		])
	)

	model = TC_MTLR(train_dataset, learning_rate=0.001)
	print("Training...")
	model.train(train_dataset, batch_size=2, num_epochs=10)
	print("Evaluating...")
	isds, cindex, ibs, mae_uncensored, mae_hinge, maepo = model.eval(train_dataset, test_dataset)
	print("Results:")
	print(f"\tC-Index: {cindex}\n\tIBS: {ibs}\n\tMAE-Uncensored: {mae_uncensored}\n\tMAE-Hinge: {mae_hinge}\n\tMAE-PO: {maepo}")