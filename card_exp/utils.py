import numpy as np
import torch
import json
from torch.autograd import Variable
import time
from mscn.util import *

def unnormalize_torch(vals, min_val, max_val):
	vals = (vals * (max_val - min_val)) + min_val
	return torch.exp(vals)


def qerror_loss(preds, targets, min_val, max_val, reduction=True):
	qerror = []
	preds = unnormalize_torch(preds, min_val, max_val)
	targets = unnormalize_torch(targets, min_val, max_val)

	for i in range(len(targets)):
		if (preds[i] > targets[i]).cpu().data.numpy()[0]:
			qerror.append(preds[i] / targets[i])
		else:
			qerror.append(targets[i] / preds[i])

	if reduction:
		return torch.mean(torch.cat(qerror))
	else:
		return torch.cat(qerror)


def predict(model, data_loader, cuda, min_val, max_val):
	preds = []
	labels = []
	t_total = 0.

	model.eval()
	for batch_idx, data_batch in enumerate(data_loader):

		samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks = data_batch
		labels.extend(targets.float())
		if cuda:
			samples, predicates, joins, targets = samples.cuda(), predicates.cuda(), joins.cuda(), targets.cuda()
			sample_masks, predicate_masks, join_masks = sample_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()
		samples, predicates, joins, targets = Variable(samples), Variable(predicates), Variable(joins), Variable(
			targets)
		sample_masks, predicate_masks, join_masks = Variable(sample_masks), Variable(predicate_masks), Variable(
			join_masks)

		t = time.time()
		outputs = model(samples, predicates, joins, sample_masks, predicate_masks, join_masks)
		t_total += time.time() - t

		for i in range(outputs.data.shape[0]):
			preds.append(np.squeeze(outputs.data[i].cpu().detach().numpy()))

	preds_test_unnorm = unnormalize_labels(preds, min_val, max_val)
	labels_test_unnorm = unnormalize_labels(labels, min_val, max_val)

	qerrors = get_qerrors(preds_test_unnorm, labels_test_unnorm)
	return preds, t_total, qerrors


def get_qerrors(preds_unnorm, labels_unnorm):
	qerror = []
	for i in range(len(preds_unnorm)):
		if preds_unnorm[i] > float(labels_unnorm[i]):
			qerror.append(preds_unnorm[i] / float(labels_unnorm[i]))
		else:
			qerror.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))
	return qerror


def print_qerror(preds_unnorm, labels_unnorm):
	qerror = []
	for i in range(len(preds_unnorm)):
		if preds_unnorm[i] > float(labels_unnorm[i]):
			qerror.append(preds_unnorm[i] / float(labels_unnorm[i]))
		else:
			qerror.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))

	print("Median: {}".format(np.median(qerror)))
	print("90th percentile: {}".format(np.percentile(qerror, 90)))
	print("95th percentile: {}".format(np.percentile(qerror, 95)))
	print("99th percentile: {}".format(np.percentile(qerror, 99)))
	print("Max: {}".format(np.max(qerror)))
	print("Mean: {}".format(np.mean(qerror)))


def get_results_by_tile(all_performs, tile='mean'):
	### get the avg qerrors of given tile at all shifting points

	avg_qerror_per_point = []
	for res_per_point in all_performs:
		avg_qerror_per_task = []
		for res_per_task in res_per_point:
			desired_qerror = res_per_task[tile]
			avg_qerror_per_task.append(desired_qerror)
		avg_qerror_per_point.append(np.mean(avg_qerror_per_task))
	avg_qerror = np.mean(avg_qerror_per_point)

	return avg_qerror


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

	file_name = "./card_result_imb_{}.txt".format(str(is_imbalance))
	with open(file_name, "a") as f:
		f.write("{}\n".format(json.dumps(result_json)))
		f.flush()