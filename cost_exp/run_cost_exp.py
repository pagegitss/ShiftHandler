import argparse
import time
import os
import numpy as np
import torch
import json

import sys

from utils import *
sys.path.insert(0, './bao_server')
import bao_server.model as model
import copy
import random

sys.path.append('../ShiftHandler')
from replay_buffer import summarizer


def train_and_predict(x, y, train_list, test_list, cuda, buffer, buffer_size, num_tasks=6, concentration=1e-4,
                      tradeoff=0.5, seed=0):
	# Load training and validation data
	print("buffer: {}".format(buffer))
	print("seed: {}".format(seed))
	print("concentration: {}".format(concentration))
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)

	reg = model.BaoRegression(have_cache_data=True, verbose=False)
	reg.fit_feature_extractor(x, y)

	all_performs = []

	latest_buffer = []
	if buffer == 'lwp':
		handler_buffer = summarizer(buffer_limit=buffer_size, loss_ada=True,
		                         concentration=concentration,
		                         is_move=False)
	else:
		handler_buffer = summarizer(buffer_limit=buffer_size, loss_ada=False,
		                         concentration=concentration,
		                         is_move=False)
	num_queries_seen_far = 0

	for task_id in range(num_tasks):
		# first replay old queries
		replay_list = []
		if buffer == 'cbp' or buffer.lower() == 'lwp':
			replay_queries_tmp, _ = handler_buffer.get_all_samples()
			for (_, y_i, plan, _, _) in replay_queries_tmp:
				replay_list.append((plan, y_i))
		elif buffer == 'latest':
			replay_list = latest_buffer
		elif buffer == 'rs':
			replay_list = latest_buffer
		elif buffer == 'all':
			replay_list = latest_buffer

		(x_train, y_train) = train_list[task_id]

		current_bs = len(x_train)
		x_train_all = copy.deepcopy(x_train)
		y_train_all = copy.deepcopy(y_train)

		for (plan, y_i) in replay_list:
			x_train_all.append(plan)
			y_train_all.append(y_i)

		ada_size = False
		if buffer.lower() == 'lwp':
			ada_size = True

		if tradeoff != 0 and buffer != 'latest' and len(replay_list) > 0:
			idx_list, current_losses, replay_idx_list, replay_losses_list = reg.fit_model(x_train_all, y_train_all,
			                                                                              tradeoff=tradeoff, ada_size=ada_size,
			                                                                              size_current_batch=current_bs, seed=seed)
		else:
			idx_list, current_losses, replay_idx_list, replay_losses_list = reg.fit_model(x_train_all, y_train_all, seed=seed,
			                                                                              ada_size=ada_size)

			# update buffer size based on loss
		if buffer == 'lwp' and len(replay_losses_list):
			norm_losses_list = [None] * handler_buffer.buffer_size
			for (q_id, loss) in zip(replay_idx_list, replay_losses_list):
				norm_losses_list[q_id] = loss[0]
			handler_buffer.update_losses(norm_losses_list)

		if task_id > 1:
			prev_perform_this_task = []
			for prev_task in range(3):
				(x_test, y_test) = test_list[prev_task]
				preds = reg.predict(x_test)
				preds = np.squeeze(preds)
				square_error = sle(preds, y_test)
				res = {"buffer": buffer, "size": buffer_size, "concentration": concentration, "seed": seed,
				       "median": np.percentile(square_error, 50), "95": np.percentile(square_error, 95),
				       "max": np.max(square_error), "mean": np.mean(square_error)}
				prev_perform_this_task.append(res)

			all_performs.append(prev_perform_this_task)

		if task_id == num_tasks - 1:
			break

		# add new queries to the replay buffer
		(x_train, y_train) = train_list[task_id]
		if buffer == 'cbp' or buffer.lower() == 'lwp':
			experience_features = reg.get_before_features(x_train)
			for i in range(experience_features.shape[0]):
				q_feature = experience_features[i, :]
				if buffer == 'lwp':
					loss_id = idx_list.index(i)
					handler_buffer.process_a_query(q_feature,  y_train[i], x_train[i], None, current_losses[loss_id][0])
				else:
					handler_buffer.process_a_query(q_feature, y_train[i], x_train[i])
		elif buffer == 'latest':
			for i in range(len(x_train)):
				if len(latest_buffer) < buffer_size:
					latest_buffer.append((x_train[i], y_train[i]))
				else:
					latest_buffer.pop(0)
					latest_buffer.append((x_train[i], y_train[i]))
		elif buffer == 'rs':
			for i in range(len(x_train)):
				if len(latest_buffer) < buffer_size:
					latest_buffer.append((x_train[i], y_train[i]))
				else:
					random_i = random.uniform(0, 1)
					if random_i < float(len(latest_buffer)) / (num_queries_seen_far + 1):
						latest_buffer.pop(0)
						latest_buffer.append((x_train[i], y_train[i]))
				num_queries_seen_far += 1
		elif buffer == 'all':
			for i in range(len(x_train)):
				latest_buffer.append((x_train[i], y_train[i]))

	mean_error = get_results_by_tile(all_performs, 'mean')
	median_error = get_results_by_tile(all_performs, 'median')
	max_error = get_results_by_tile(all_performs, 'max')

	return mean_error, median_error, max_error


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--imbalance', default=False, help="is imbalance?", action='store_true')
	parser.add_argument("--buffersize", help="buffer size", type=int, default=50)

	parser.add_argument("--epochs", help="number of epochs (default: 20)", type=int, default=30)
	parser.add_argument("--batch", help="batch size (default: 1024)", type=int, default=1024)
	parser.add_argument("--cuda", help="use CUDA", action="store_true")
	args = parser.parse_args()

	concentrations = [1e-2, 1e-1, 1.0, 10, 100]
	random_seeds = list(range(10))
	is_imb = args.imbalance

	f_train = open("./plans.txt", 'r')
	lines = f_train.readlines()
	x = []
	y = []
	for line in lines:
		line = line.strip()
		parts = line.split('|')
		y_i = parts[-1]
		x_i = "".join(parts[:len(parts) - 1])

		x.append(x_i)
		y.append(float(y_i))

	num_burnin = 2
	num_tasks = num_burnin + 3

	### init the dics for storing results
	result_per_seed_all = {}
	result_per_seed_rs = {}
	result_per_seed_latest = {}
	result_per_seed_lwp = {}
	result_per_seed_cbp = {}

	for random_seed in random_seeds:
		x_f, y_f, train_list, test_list = get_query_list(x, y, num_train_per_task=200, num_test_per_task=100, is_imb=is_imb,
		                                                 num_burnin=num_burnin,
		                                                 seed=random_seed)

		### exp of ALL
		mean_error, median_error, max_error = train_and_predict(x_f, y_f, train_list, test_list, args.cuda, 'all',
		                                                        args.buffersize, num_tasks=num_tasks, seed=random_seed)
		result_per_seed_all[random_seed] = {'mean':mean_error, 'median':median_error, 'max':max_error}

		### exp of random sampling
		mean_error, median_error, max_error = train_and_predict(x_f, y_f, train_list, test_list, args.cuda, 'rs',
		                  args.buffersize, num_tasks=num_tasks, seed=random_seed)
		result_per_seed_rs[random_seed] = {'mean':mean_error, 'median':median_error, 'max':max_error}

		### exp of latest
		mean_error, median_error, max_error = train_and_predict(x_f, y_f, train_list, test_list, args.cuda, 'latest',
		                  args.buffersize, num_tasks=num_tasks, seed=random_seed)
		result_per_seed_latest[random_seed] = {'mean':mean_error, 'median':median_error, 'max':max_error}

		### exp of cbp
		mean_error_per_concentration = {}
		median_error_per_concentration = {}
		max_error_per_concentration = {}
		for concentration in concentrations:
			mean_error, median_error, max_error = train_and_predict(x_f, y_f, train_list, test_list, args.cuda, 'cbp',
			                  args.buffersize, concentration=concentration, num_tasks=num_tasks, seed=random_seed)
			mean_error_per_concentration[concentration] = mean_error
			median_error_per_concentration[concentration] = median_error
			max_error_per_concentration[concentration] = max_error
		best_c = min(mean_error_per_concentration, key=mean_error_per_concentration.get)
		result_per_seed_cbp[random_seed] = {'mean': mean_error_per_concentration[best_c],
		                                    'median': median_error_per_concentration[best_c],
		                                    'max': max_error_per_concentration[best_c]}

		### exp of lwp
		mean_error_per_concentration = {}
		median_error_per_concentration = {}
		max_error_per_concentration = {}
		for concentration in concentrations:
			mean_error, median_error, max_error = train_and_predict(x_f, y_f, train_list, test_list, args.cuda, 'lwp',
			                  args.buffersize, concentration=concentration, num_tasks=num_tasks, seed=random_seed)
			mean_error_per_concentration[concentration] = mean_error
			median_error_per_concentration[concentration] = median_error
			max_error_per_concentration[concentration] = max_error
		best_c = min(mean_error_per_concentration, key=mean_error_per_concentration.get)
		result_per_seed_lwp[random_seed] = {'mean': mean_error_per_concentration[best_c],
		                                    'median': median_error_per_concentration[best_c],
		                                    'max': max_error_per_concentration[best_c]}

	### start writing the results
	write_results('all', args.buffersize, is_imb, result_per_seed_all)
	write_results('rs', args.buffersize, is_imb, result_per_seed_rs)
	write_results('latest', args.buffersize, is_imb, result_per_seed_latest)
	write_results('cbp', args.buffersize, is_imb, result_per_seed_cbp)
	write_results('lwp', args.buffersize, is_imb, result_per_seed_lwp)



if __name__ == "__main__":
	main()
