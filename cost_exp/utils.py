import numpy as np
import torch
import json
from torch.autograd import Variable
import time
import random
import csv
import copy
from random import shuffle

def unnormalize_torch(vals, min_val, max_val):
	vals = (vals * (max_val - min_val)) + min_val
	return torch.exp(vals)


def get_query_list(x, y, num_train_per_task, num_test_per_task, is_imb=True, num_burnin=4, num_task=3, seed=0):
	file_name = "./train"
	num_queries_per_file = 100000

	x_filtered = []
	y_filtered = []

	np.random.seed(seed)
	random.seed(seed)

	joins = []
	predicates = []
	tables = []
	label = []
	numerical_label = []

	templates = {}
	template_joins = {}
	template_ids = []
	temp_counts = []

	with open(file_name + ".csv", 'rU') as f:
		data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
		for row in data_raw[:num_queries_per_file]:
			tables.append(row[0].split(','))
			joins.append(row[1].split(','))
			predicates.append(row[2].split(','))

			joined_tables = row[0].split(',')
			joined_tables.sort()
			if ','.join(joined_tables) not in templates:
				templates[','.join(joined_tables)] = len(templates)
				template_joins[templates[','.join(joined_tables)]] = len(joined_tables)
				temp_counts.append(0)
			template_ids.append(templates[','.join(joined_tables)])
			temp_counts[templates[','.join(joined_tables)]] += 1

			if int(row[3]) < 1:
				print("Queries must have non-zero cardinalities")
				exit(1)
			label.append(row[3])
			numerical_label.append(int(row[3]))
	print("Loaded queries")

	if is_imb:
		ratio_per_temp = [5, 20, 75]
	else:
		ratio_per_temp = [33, 33, 34]

	ratio_per_temp.reverse()

	filtered_temp_list_all = [i for i in range(len(temp_counts)) if temp_counts[i] >= 1100]
	filtered_temp_list = []

	mean_list = []
	for tmp_id in filtered_temp_list_all:
		total_id_per_tmp = [i for i in range(num_queries_per_file) if template_ids[i] == tmp_id]
		y_list = []
		for idx in total_id_per_tmp:
			y_list.append(numerical_label[idx])
		mean_list.append(np.mean(y_list))

	median_list = []
	for tmp_id in filtered_temp_list_all:
		total_id_per_tmp = [i for i in range(num_queries_per_file) if template_ids[i] == tmp_id]
		y_list = []
		for idx in total_id_per_tmp:
			y_list.append(numerical_label[idx])
		median_list.append(np.median(y_list))

	sort_index = [i for i, x in sorted(enumerate(median_list), key=lambda x: x[1])]
	num_per_category = int(len(sort_index)/num_task)
	task_candi_lens = [6, 8, 5]

	for i in range(num_task):
		task_candi = sort_index[sum(task_candi_lens[:i]):sum(task_candi_lens[:i+1])]
		chosen_idx = random.choice(task_candi)
		filtered_temp_list.append(filtered_temp_list_all[chosen_idx])
	median_list.sort()
	train_list = []
	test_list = []

	candi_train_id_list = []
	candi_test_id_list = []


	for ind_i, ratio_i in enumerate(ratio_per_temp):
		temp_id = filtered_temp_list[ind_i]
		total_id_per_tmp = [i for i in range(num_queries_per_file) if template_ids[i] == temp_id]
		shuffle(total_id_per_tmp)
		# total_id_per_tmp = total_id_per_tmp[0:-num_test_per_task]
		candi_train_id_list.append(total_id_per_tmp[0:-num_test_per_task])
		candi_test_id_list.append(total_id_per_tmp[-num_test_per_task:])

	x_train_tmp = []
	y_train_tmp = []
	for _ in range(num_burnin):
		for ind_i, ratio_i in enumerate(ratio_per_temp):
			num_queries_train_per_temp = int(ratio_per_temp[ind_i] * num_train_per_task / sum(ratio_per_temp))
			total_id_per_tmp = copy.deepcopy(candi_train_id_list[ind_i])
			shuffle(total_id_per_tmp)
			train_id_per_tmp = total_id_per_tmp[:num_queries_train_per_temp]

			x_train_tmp.extend([x[q_i] for q_i in train_id_per_tmp])
			y_train_tmp.extend([y[q_i] for q_i in train_id_per_tmp])

			if num_queries_train_per_temp > len(total_id_per_tmp):
				k_needed = num_queries_train_per_temp - len(total_id_per_tmp)
				sampled_per_tmp = random.choices(total_id_per_tmp, k=k_needed)
				x_train_addition = [x[q_i] for q_i in sampled_per_tmp]
				y_train_addition = [y[q_i] for q_i in sampled_per_tmp]

				x_train_tmp.extend(x_train_addition)
				y_train_tmp.extend(y_train_addition)

	x_filtered.extend(x_train_tmp)
	y_filtered.extend(y_train_tmp)

	for i in range(num_burnin):
		train_list.append((x_train_tmp[i * num_train_per_task: i * num_train_per_task + num_train_per_task],
		                   y_train_tmp[i * num_train_per_task: i * num_train_per_task + num_train_per_task]))

	for task_id in range(len(ratio_per_temp)):

		train_id_per_tmp = copy.deepcopy(candi_train_id_list[task_id])
		shuffle(train_id_per_tmp)
		train_id_per_tmp = train_id_per_tmp[:num_train_per_task]

		test_id_per_tmp = copy.deepcopy(candi_test_id_list[task_id])

		x_train = [x[q_i] for q_i in train_id_per_tmp]
		y_train = [y[q_i] for q_i in train_id_per_tmp]
		train_list.append((x_train, y_train))

		x_filtered.extend(x_train)
		y_filtered.extend(y_train)

		x_test = [x[q_i] for q_i in test_id_per_tmp]
		y_test = [y[q_i] for q_i in test_id_per_tmp]
		test_list.append((x_test, y_test))

		x_filtered.extend(x_test)
		y_filtered.extend(y_test)
	return x_filtered, y_filtered, train_list, test_list


def sle(preds, targets):
	log_preds = np.log(preds)
	log_targets = np.log(targets)

	min_val = np.min(log_targets)
	max_max = np.max(log_targets)

	scaled_preds = [(pred - min_val) / (max_max - min_val) for pred in log_preds]
	scaled_targets = [(pred - min_val) / (max_max - min_val) for pred in log_targets]

	sle_res = []
	for i in range(len(targets)):
		err = np.square(scaled_preds[i] - scaled_targets[i])
		sle_res.append(err)
	return sle_res


def get_results_by_tile(all_performs, tile='mean'):
	### get the avg errors of given tile at all shifting points

	avg_error_per_point = []
	for res_per_point in all_performs:
		avg_error_per_task = []
		for res_per_task in res_per_point:
			desired_error = res_per_task[tile]
			avg_error_per_task.append(desired_error)
		avg_error_per_point.append(np.mean(avg_error_per_task))
	avg_error = np.mean(avg_error_per_point)

	return avg_error

def write_results(buffer, buffer_size, is_imbalance, result_per_seed):
	avg_mean, avg_median, avg_max = 0, 0, 0
	for seed in result_per_seed:
		avg_mean += result_per_seed[seed]['mean']
		avg_median += result_per_seed[seed]['median']
		avg_max += result_per_seed[seed]['max']

	avg_mean = avg_mean / len(result_per_seed)
	avg_median = avg_median / len(result_per_seed)
	avg_max = avg_max / len(result_per_seed)

	overall_performance = {'mean': avg_mean, 'median': avg_median, 'max': avg_max}

	result_json = {"buffer": buffer, "size": buffer_size, "overall_performance": overall_performance, "result_per_seed": result_per_seed}

	file_name = "./cost_result_imb_{}.txt".format(str(is_imbalance))
	with open(file_name, "a") as f:
		f.write("{}\n".format(json.dumps(result_json)))
		f.flush()
