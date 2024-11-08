import random
import sys
import numpy as np
import json
from scipy import spatial


class summarizer:
	def __init__(self, buffer_limit=300, concentration=1., loss_ada=False, check_duplicate=False,
	             y_gap=30000000., is_move=False, tradeoff=0.5, seed=0):

		self.concentration = concentration
		self.loss_ada = loss_ada
		self.y_gap = y_gap
		self.is_move = is_move
		self.tradeoff = tradeoff
		self.seed = seed
		self.rs = np.random.RandomState(seed)
		self.buffer_limit = buffer_limit
		self.is_print = False
		self.check_duplicate = check_duplicate
		self.last_e_id = -1

		# for K-Medoids
		self.centers = []

		self.buffer_size = 0  # current buffer size

		# cluster memberships
		self.id2query = {}  # mapping from id to actual query
		self.ids_per_cluster = []  # the ids of queries
		self.cluster_centers = []  # cluster centers
		self.kd_tree = None
		self.ever_num_cluster = 0

		# running statistics on cluster level
		self.cluster_num_noncenter = []  # number of non-centers in each cluster
		self.cluster_ever_num_noncenter = []  # number of non-centers ever assigned to each cluster

		self.radius_per_cluster = []

		np.random.seed(self.seed)
		random.seed(self.seed)

	def get_x_dim(self):
		x_1 = self.id2query[next(iter(self.id2query))][0]
		return x_1.shape[0]

	def remove_noncenter_from_cluster(self, cluster_id, q_id=None):
		qids_in_cluster = self.ids_per_cluster[cluster_id]
		center_qid = self.cluster_centers[cluster_id]
		noncenters = [q_id for q_id in qids_in_cluster if q_id != center_qid]

		if q_id is None:
			removed_qid = self.rs.choice(noncenters)
		else:
			removed_qid = q_id

		self.ids_per_cluster[cluster_id].remove(removed_qid)
		self.buffer_size -= 1

		self.cluster_num_noncenter[cluster_id] -= 1

		del self.id2query[removed_qid]

	def remove_a_cluster(self, cluster_id):
		qids_in_cluster = self.ids_per_cluster[cluster_id]
		center_qid = self.cluster_centers[cluster_id]
		noncenters = [q_id for q_id in qids_in_cluster if q_id != center_qid]

		total_size = len(noncenters) + 1

		# first delete noncenters
		for noncenter_qid in noncenters:
			del self.id2query[noncenter_qid]

		# then delete the center
		del self.ids_per_cluster[cluster_id]
		self.cluster_centers.remove(center_qid)

		self.buffer_size -= total_size

		del self.cluster_num_noncenter[cluster_id]
		del self.cluster_ever_num_noncenter[cluster_id]
		del self.radius_per_cluster[cluster_id]
		del self.id2query[center_qid]

		self.ever_num_cluster -= 1

	def remove_center(self, c_id=None):
		num_clusters = len(self.cluster_centers)
		if c_id is None:
			removed_cid = self.rs.choice(range(num_clusters))
		else:
			removed_cid = c_id
		removed_qid = self.cluster_centers[removed_cid]

		del self.ids_per_cluster[removed_cid]
		self.cluster_centers.remove(removed_qid)

		self.buffer_size -= 1

		del self.cluster_num_noncenter[removed_cid]
		del self.cluster_ever_num_noncenter[removed_cid]
		del self.id2query[removed_qid]

	def remove_sample(self, removed_qid):
		centers = self.cluster_centers
		if removed_qid in centers:
			# is a center
			removed_cid = centers.index(removed_qid)
			if self.cluster_num_noncenter[removed_cid] == 0:
				# only one query in the cluster
				del self.ids_per_cluster[removed_cid]
				self.cluster_centers.remove(removed_qid)

				self.buffer_size -= 1
				self.ever_num_cluster -= 1

				del self.cluster_num_noncenter[removed_cid]
				del self.cluster_ever_num_noncenter[removed_cid]
				del self.radius_per_cluster[removed_cid]
				del self.id2query[removed_qid]
			else:
				# there are noncenters in the cluster
				remainings = []

				for qid in self.ids_per_cluster[removed_cid]:
					if qid != removed_qid:
						remainings.append(self.id2query[qid])

				self.remove_a_cluster(removed_cid)

				# replay the remaining samples but the center removed
				for remain_query in remainings:
					self.do_a_query(remain_query[0], remain_query[1], remain_query[2], remain_query[3], remain_query[4])
			self.update_kdtree()
		else:
			# not a center
			cluster_id = 0
			for tmp_cluster_id, samples_in_center in enumerate(self.ids_per_cluster):
				if removed_qid in samples_in_center:
					cluster_id = tmp_cluster_id
					break

			self.ids_per_cluster[cluster_id].remove(removed_qid)
			self.buffer_size -= 1

			self.cluster_num_noncenter[cluster_id] -= 1
			self.cluster_ever_num_noncenter[cluster_id] -= 1

			del self.id2query[removed_qid]

	def update_kdtree(self):
		A = []
		for q_i in self.cluster_centers:
			A.append(self.id2query[q_i][0])
		self.kd_tree = spatial.KDTree(np.array(A))

	def get_cluster_losses(self, type='all'):
		losses = []
		cluster_ids = []
		if type == 'all':
			for c_id, clusters in enumerate(self.ids_per_cluster):
				loss_in_cluster = 0.
				for q_id in clusters:
					loss_in_cluster += self.id2query[q_id][4]
				losses.append(loss_in_cluster)
				cluster_ids.append(c_id)
		elif type == 'noncenter':
			for c_id, clusters in enumerate(self.ids_per_cluster):
				loss_in_cluster = 0.
				if len(clusters) > 1:
					for q_id in clusters:
						loss_in_cluster += self.id2query[q_id][4]
					losses.append(loss_in_cluster)
					cluster_ids.append(c_id)
		return losses, cluster_ids

	def loss_assign_center(self, x, y, new_radius=None, removed_centers=None, info=None, tem_id=None, loss=None):
		if self.is_print:
			print("running loss_assign_center")
		# first try to remove a non-center
		noncluster_size = self.cluster_num_noncenter

		if max(noncluster_size) > 0:
			# there are noncenters
			losses, sample_cluster_ids = self.get_cluster_losses(type='noncenter')
			weights = [1./l if l > 0 else 0. for l in losses]
			weights = [w/sum(weights) for w in weights]
			random_cluster_id = self.rs.choice(sample_cluster_ids, p=weights)

			self.remove_noncenter_from_cluster(random_cluster_id)
			self.add_sample(x, y, True, -1,
			                new_radius=new_radius, removed_centers=removed_centers, info=info, tem_id=tem_id, loss=loss)
		else:
			# there is no noncenter, we must remove a cluster
			losses, sample_cluster_ids = self.get_cluster_losses(type='all')
			losses.append(loss)
			weights = [1. / l if l > 0 else 0. for l in losses]
			weights = [w / sum(weights) for w in weights]
			sample_cluster_ids.append(-1)
			random_cluster_id = self.rs.choice(sample_cluster_ids, p=weights)

			if random_cluster_id != -1:
				self.remove_center(random_cluster_id)
				self.add_sample(x, y, True, -1,
				                new_radius=new_radius, removed_centers=removed_centers, info=info, tem_id=tem_id, loss=loss)

			# else just ignore the query

	def loss_assign_noncenter(self, x, y, cluster_id, new_radius=None, removed_centers=None, info=None, tem_id=None, loss=None):
		# find out max clusters
		if self.is_print:
			print("running loss_assign_noncenter")

		self.add_sample(x, y, False, cluster_id,
		                new_radius=new_radius, removed_centers=removed_centers, info=info, tem_id=tem_id, loss=loss)
		noncluster_size = self.cluster_num_noncenter

		if max(noncluster_size) > 0:
			# there are noncenters
			losses, sample_cluster_ids = self.get_cluster_losses(type='noncenter')
			weights = [1./l if l > 0 else 0. for l in losses]
			weights = [w/sum(weights) for w in weights]
			random_cluster_id = self.rs.choice(sample_cluster_ids, p=weights)
			self.remove_noncenter_from_cluster(random_cluster_id)
		else:
			# there is no noncenter, we must remove a cluster
			losses, sample_cluster_ids = self.get_cluster_losses(type='all')
			weights = [1. / l if l > 0 else 0. for l in losses]
			weights = [w / sum(weights) for w in weights]
			random_cluster_id = self.rs.choice(sample_cluster_ids, p=weights)
			self.remove_center(random_cluster_id)

	def rs_assign_center(self, x, y, new_radius=None, removed_centers=None, info=None, tem_id=None, loss=None):
		if self.is_print:
			print("running rs_assign_center")
		# first try to remove a non-center
		noncluster_size = self.cluster_num_noncenter

		if max(noncluster_size) > 0:
			# there are noncenters
			max_clusters = [index for index, item in enumerate(noncluster_size) if item == max(noncluster_size)]
			random_cluster_id = self.rs.choice(max_clusters)
			self.remove_noncenter_from_cluster(random_cluster_id)
			self.add_sample(x, y, True, -1,
			                new_radius=new_radius, removed_centers=removed_centers, info=info, tem_id=tem_id, loss=loss)
		else:
			# there is no noncenter, we must remove a cluster
			num_clusters = len(self.cluster_centers)
			i = self.rs.uniform()
			if i < (float(num_clusters) / self.ever_num_cluster):
				self.remove_center()
				self.add_sample(x, y, True, -1,
				                new_radius=new_radius, removed_centers=removed_centers, info=info, tem_id=tem_id, loss=loss)

			# else just ignore the query

	def rs_assign_noncenter(self, x, y, cluster_id, new_radius=None, removed_centers=None, info=None, tem_id=None, loss=None):
		# find out max clusters
		if self.is_print:
			print("running rs_assign_noncenter")
		noncluster_size = self.cluster_num_noncenter
		max_clusters = [index for index, item in enumerate(noncluster_size) if item == max(noncluster_size)]
		if cluster_id not in max_clusters:
			# not one of the max clusters
			random_cluster_id = self.rs.choice(max_clusters)
			self.remove_noncenter_from_cluster(random_cluster_id)
			self.add_sample(x, y, False, cluster_id,
			                new_radius=new_radius, removed_centers=removed_centers, info=info, tem_id=tem_id, loss=loss)
		else:
			# is one of the max clusters
			num_clusters = len(self.cluster_centers)
			if (self.buffer_size - num_clusters) > 0:
				# there are non-center queries
				i = self.rs.uniform()
				if i < (float(self.cluster_num_noncenter[cluster_id]) / float(
						self.cluster_ever_num_noncenter[cluster_id])):
					self.remove_noncenter_from_cluster(cluster_id)
					self.add_sample(x, y, False, cluster_id,
					                new_radius=new_radius, removed_centers=removed_centers, info=info,
					                tem_id=tem_id, loss=loss)

	def add_sample(self, x, y, is_newcenter, cluster_id,
	               new_radius=None, removed_centers=None, info=None, tem_id=None, loss=None):
		q_id = 0
		# first we need to find a proper id for the new query
		is_found = False
		for i in range(self.buffer_size):
			if i not in self.id2query:
				q_id = i
				is_found = True
		if not is_found:
			q_id = len(self.id2query)

		self.id2query[q_id] = (x, y, info, tem_id, loss)

		self.buffer_size += 1

		if not is_newcenter:
			self.ids_per_cluster[cluster_id].append(q_id)
			self.cluster_num_noncenter[cluster_id] += 1
			self.cluster_ever_num_noncenter[cluster_id] += 1

		else:
			# for the case of new center
			new_cluster_id = len(self.ids_per_cluster)
			self.ids_per_cluster.append([q_id])
			self.cluster_centers.append(q_id)
			self.ever_num_cluster += 1
			self.cluster_num_noncenter.append(0.)
			self.cluster_ever_num_noncenter.append(0.)

			self.update_kdtree()

			if new_radius is not None:
				self.radius_per_cluster.append(new_radius)
			else:
				self.radius_per_cluster.append(self.concentration)

			if removed_centers is not None:
				# first add queries to the new cluster
				for cluster_id in removed_centers:
					for q_id in self.ids_per_cluster[cluster_id]:
						self.ids_per_cluster[new_cluster_id].append(q_id)
						self.cluster_num_noncenter[new_cluster_id] += 1
						self.cluster_ever_num_noncenter[new_cluster_id] += 1

				# then delete old clusters
				removed_centers.reverse()
				for removed_cid in removed_centers:
					removed_qid = self.cluster_centers[removed_cid]

					del self.ids_per_cluster[removed_cid]
					self.cluster_centers.remove(removed_qid)
					del self.cluster_num_noncenter[removed_cid]
					del self.cluster_ever_num_noncenter[removed_cid]
					del self.radius_per_cluster[removed_cid]
				self.update_kdtree()

	def update_losses(self, losses):
		count = 0
		for c_id, clusters in enumerate(self.ids_per_cluster):
			for q_id in clusters:
				self.id2query[q_id] = (self.id2query[q_id][0], self.id2query[q_id][1], self.id2query[q_id][2],
				                       self.id2query[q_id][3], losses[count])
				count += 1

	def ODMedoids(self, x, y):
		# return three values:
		# 1. If increase the buffer size by 1 (always True in ODMdoids)
		# 2. If (x, y) is a new center
		# 3. The center id of (x, y) if not a new center; otherwise return a dumb value
		# 4. The minimal distance

		if self.is_print:
			print("running ODMedoids for {} ".format(x,))

		center_q_ids = self.cluster_centers

		# no cluster
		if len(center_q_ids) == 0:
			return True, True, -1, 0

		min_d, nearest_cluster = self.kd_tree.query(x)
		min_d = np.square(min_d)
		center_q_id = center_q_ids[nearest_cluster]

		min_y_i = self.id2query[center_q_id][1]

		i = self.rs.uniform()
		if self.is_print:
			print("min_d={}; concentration={}, randon sample={}".format(min_d, self.concentration, i))
		if i < (min_d / self.concentration):
			# q is a new center
			if self.is_print:
				print("so we create a new cluster for this query")
			return True, True, -1, min_d
		elif np.abs(min_y_i - y) < self.y_gap:
			if self.is_print:
				print("so we assign the query to the nearest cluster")
			return True, False, nearest_cluster, min_d
		else:
			if self.is_print:
				print("so we create a new cluster for this query")
			return True, True, -1, min_d

	def ODMedoids_CM(self, x, y):
		# return three values:
		# 1. If increase the buffer size by 1
		# 2. If (x, y) is a new center
		# 3. The center id of (x, y) if not a new center; otherwise return a dumb value
		# 4. The minimal distance

		center_q_ids = self.cluster_centers

		if len(center_q_ids) == 0:
			return True, True, -1, 0, self.concentration, None

		nearest_cluster = 0
		min_d = 0
		for cluster_id, q_id in enumerate(center_q_ids):
			u_i = self.id2query[q_id][0]
			d_i = np.square(np.linalg.norm(u_i - x)) + 2 * self.radius_per_cluster[cluster_id]
			if cluster_id == 0:
				min_d = d_i
			else:
				if d_i < min_d:
					min_d = d_i
					nearest_cluster = cluster_id

		i = self.rs.uniform()

		if i < (min_d / self.concentration):
			# q is a new center
			new_radius = min([min_d, self.concentration]) / 6.0
			removed_centers = []

			for cluster_id, q_id in enumerate(center_q_ids):
				u_i = self.id2query[q_id][0]
				if np.square(np.linalg.norm(u_i - x)) <= self.radius_per_cluster[cluster_id]:
					removed_centers.append(cluster_id)
			return True, True, -1, min_d, new_radius, removed_centers
		else:
			return True, False, nearest_cluster, min_d, None, None

	def do_a_query(self, x, y, info=None, tem_id=None, loss=None):
		if not self.is_move:
			# the case for ODMedoids
			is_increase, is_newcenter, cluster_id, min_d = self.ODMedoids(x, y)
			removed_centers = None
			new_radius = None
		else:
			# the case for ODMedoids-CM
			is_increase, is_newcenter, cluster_id, min_d, new_radius, removed_centers = self.ODMedoids_CM(x, y)

		##########
		# 1. for center, add new samples (possibly add and merge clusters) by add_sample,
		# 2. for non-center, check if the cluster limit is reached, if so, remove a noncenter via rs
		if self.buffer_size + 1 > self.buffer_limit:
			if is_newcenter:
				if self.loss_ada:
					self.loss_assign_center(x, y, new_radius=new_radius, removed_centers=removed_centers, info=info,
					                      tem_id=tem_id, loss=loss)
				else:
					self.rs_assign_center(x, y, new_radius=new_radius, removed_centers=removed_centers, info=info,
							                tem_id=tem_id, loss=loss)
			else:
				if self.loss_ada:
					self.loss_assign_noncenter(x, y, cluster_id, new_radius=new_radius,
				                         removed_centers=removed_centers, info=info, tem_id=tem_id, loss=loss)
				else:
					self.rs_assign_noncenter(x, y, cluster_id, new_radius=new_radius,
					                         removed_centers=removed_centers, info=info, tem_id=tem_id, loss=loss)

		else:
			# buffer limit is not reached
			self.add_sample(x, y, is_newcenter, cluster_id,
			                new_radius=new_radius, removed_centers=removed_centers, info=info, tem_id=tem_id, loss=loss)

	def process_a_query(self, x, y, info=None, tem_id=None, loss=None):
		if self.check_duplicate:
			# check if the same query is in the buffer
			for q_id_candi in self.id2query:
				if np.array_equal(self.id2query[q_id_candi][0], x):
					if info == self.id2query[q_id_candi][2]:
						return 0
					else:
						if y == self.id2query[q_id_candi][1]:
							self.id2query[q_id_candi] = (
							self.id2query[q_id_candi][0], self.id2query[q_id_candi][1], info, tem_id)
							return 0
						else:
							self.id2query[q_id_candi] = (self.id2query[q_id_candi][0], y, info, tem_id)
							return 0

				elif info == self.id2query[q_id_candi][2]:
					# remove the old one and add the new one later
					self.remove_sample(q_id_candi)
					break

		# then actually process this query
		self.do_a_query(x, y, info, tem_id, loss)

	def get_all_samples(self):
		ret = []
		cluster_ids = []

		if self.buffer_size == 0:
			return [], []

		for c_id, clusters in enumerate(self.ids_per_cluster):
			for q_id in clusters:
				ret.append(self.id2query[q_id])
				cluster_ids.append(c_id)

		return ret, cluster_ids
