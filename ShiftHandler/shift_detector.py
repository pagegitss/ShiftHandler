import random
import sys
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import ks_2samp

class shift_detector:
	def __init__(self, sign_level=0.05, reduced_dim=32, seed=0):

		self.sign_level = sign_level
		self.reduced_dim = reduced_dim
		self.seed = seed
		self.rs = np.random.RandomState(seed)

		np.random.seed(self.seed)
		random.seed(self.seed)

	def one_kv_test(self, P, Q):
		# Compute KS statistic and p-value
		t_val, p_val = ks_2samp(P, Q)
		return p_val

	def multi_kv_test(self, P, Q):
		p_vals = []

		# For each dimension we conduct a separate KS test
		for i in range(P.shape[1]):
			feature_p = P[:, i]
			feature_q = Q[:, i]

			# Compute KS statistic and p-value
			t_val, p_val = ks_2samp(feature_p, feature_q)
			p_vals.append(p_val)

		# Apply the Bonferroni correction to bound the family-wise error rate.
		p_vals = np.array(p_vals)
		p_val = min(np.min(p_vals), 1.0)

		return p_val, p_vals

	def PCA(self, x):
		pca = PCA(n_components=self.reduced_dim)
		pca.fit(x)
		return pca

	def label_monitor(self, P, Q):
		# P: [num_samples_in_batch]: 1-dimensional

		p_val = self.one_kv_test(P, Q)
		decision = 1 if p_val < self.sign_level else 0

		return decision

	def query_monitor(self, P, Q):
		p_val, p_vals = self.multi_kv_test(P, Q)
		adjust_sign_level = self.sign_level / Q.shape[1]

		decision = 1 if p_val < adjust_sign_level else 0

		return decision
