import copy
import numpy as np
import pandas as pd
from dataclasses import dataclass
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from utils import MTLRDataGenerator, get_data, train_val_test_split, median_time_bins, quantile_time_bins, convert_to_counting_process
from SurvivalEVAL.Evaluator import LifelinesEvaluator, SurvivalEvaluator
from lifelines import CoxTimeVaryingFitter

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

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

class CoxTVC(object):
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

		self.model = CoxTimeVaryingFitter(penalizer=0.1)

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

	def train(self, train):
		with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
			print(train)
		self.model.fit(train, id_col="id", event_col="event", start_col="start", stop_col="stop", show_progress=True)

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

		train_all, low_var_cols = convert_to_counting_process(X_train, ts_train, cs_train, y_train, rs_train, seqs_ts_train)
		train_initial, _ = convert_to_counting_process(X_train, ts_train, cs_train, y_train, rs_train, seqs_ts_train, all_states=False, low_var_cols=low_var_cols)

		val, _ = convert_to_counting_process(X_val, ts_val, cs_val, y_val, rs_val, seqs_ts_val, all_states=False, low_var_cols=low_var_cols)

		test, _ = convert_to_counting_process(X_test, ts_test, cs_test, y_test, rs_test, seqs_ts_test, all_states=False, low_var_cols=low_var_cols)

		return train_all, train_initial, val, test

	def eval(self, train, test):
		# cum_haz = self.model.predict_cumulative_hazard(test)
		# isds = np.exp(-cum_haz)

		ids = test["id"].unique()

		base_ch = self.model.baseline_cumulative_hazard_.iloc[:,0]

		results = {}
		for subject_id in test["id"]:
			subj_row = test[test["id"] == subject_id]
			log_risk = self.model.predict_log_partial_hazard(subj_row).iloc[0]
			risk = np.exp(log_risk)

			cumhaz = base_ch * risk
			surv = np.exp(-cumhaz)

			results[subject_id] = pd.Series(surv)

		isds = pd.DataFrame(results)          # time × subject
		self.time_bins = isds.index.to_numpy()     # correct time grid

		# 2. Set first/last survival correctly
		#isds.iloc[0, :] = 1.0
		isds.iloc[-1, :] = 0.0
		isds = isds

		train_event_times = train.stop.values
		train_event_indicators = train.event.values
		test_event_times = test.stop.values
		test_event_indicators = test.event.values

		# 3. Pass to SurvivalEvaluator directly
		evaluator = LifelinesEvaluator(
			isds,                    # DataFrame: time × subjects
			test_event_times,        # 1D float array
			test_event_indicators,   # 1D float array
			train_event_times,        # 1D float array
			train_event_indicators   # 1D float array
		)

		#evaluator = LifelinesEvaluator(isds, test["stop"], test["event"], train["stop"], train["event"])
		# evaluator = SurvivalEvaluator(isds, time_bins, test["stop"], test["event"], train["stop"], train["event"])
		
		#predicted_times = evaluator.predict_time_from_curve(evaluator.predict_time_method)

		cindex, concordant_pairs, total_pairs = evaluator.concordance(ties="None")
		ibs = evaluator.integrated_brier_score(num_points=isds.shape[1], IPCW_weighted=True, draw_figure=False)
		mae_uncensored = evaluator.mae(method='Uncensored')
		mae_hinge = evaluator.mae(method='Hinge')
		maepo = evaluator.mae(method='Pseudo_obs', weighted=True, truncated_time=np.max(self.time_bins))

		return isds, cindex, ibs, mae_uncensored, mae_hinge, maepo

		