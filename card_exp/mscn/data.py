import csv
import torch
from torch.utils.data import dataset

from mscn.util import *
from random import shuffle
import random

def load_data_both(file_name, num_materialized_samples, num_queries_per_file, reharse_queries=True,
                     num_queries_per_task=1000, num_task=5, seed=0):
    np.random.seed(seed)
    random.seed(seed)

    joins = []
    predicates = []
    tables = []
    samples = []
    label = []
    numerical_label = []

    templates = {}
    template_joins = {}
    template_ids = []
    temp_counts = []

    # Load queries
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

    # Load bitmaps
    num_bytes_per_bitmap = int((num_materialized_samples + 7) >> 3)
    with open(file_name + ".bitmaps", 'rb') as f:
        for i in range(len(tables)):
            four_bytes = f.read(4)
            if not four_bytes:
                print("Error while reading 'four_bytes'")
                exit(1)
            num_bitmaps_curr_query = int.from_bytes(four_bytes, byteorder='little')
            bitmaps = np.empty((num_bitmaps_curr_query, num_bytes_per_bitmap * 8), dtype=np.uint8)
            for j in range(num_bitmaps_curr_query):
                # Read bitmap
                bitmap_bytes = f.read(num_bytes_per_bitmap)
                if not bitmap_bytes:
                    print("Error while reading 'bitmap_bytes'")
                    exit(1)
                bitmaps[j] = np.unpackbits(np.frombuffer(bitmap_bytes, dtype=np.uint8))
            samples.append(bitmaps)
    print("Loaded bitmaps")

    # Split predicates
    predicates = [list(chunks(d, 3)) for d in predicates]

    filtered_temp_list_all = [i for i in range(len(temp_counts)) if temp_counts[i] >= 4000]
    filtered_temp_list = []

    for i in range(num_task):
        task_candi = []
        for tmp_id in filtered_temp_list_all:
            if template_joins[tmp_id] == (i % 3)+1:
                task_candi.append(tmp_id)
        filtered_temp_list.append(random.choice(task_candi))

    # shuffle the template ordering
    shuffled_task_list = list(range(num_task))
    # shuffle(shuffled_task_list)

    query_id_list = []
    num_queries_per_temp = []
    for task_id in shuffled_task_list:
        temp_id = filtered_temp_list[task_id]
        total_id_per_tmp = [i for i in range(num_queries_per_file) if template_ids[i] == temp_id]
        shuffle(total_id_per_tmp)
        total_id_per_tmp = total_id_per_tmp
        query_id_list.extend(total_id_per_tmp)
        num_queries_per_temp.append(len(total_id_per_tmp))

    if reharse_queries:
        shuffle_id_all_queries = query_id_list

        joins = [joins[x] for x in shuffle_id_all_queries]
        predicates = [predicates[x] for x in shuffle_id_all_queries]
        tables = [tables[x] for x in shuffle_id_all_queries]
        samples = [samples[x] for x in shuffle_id_all_queries]
        label = [label[x] for x in shuffle_id_all_queries]
        template_ids = [template_ids[x] for x in shuffle_id_all_queries]

    return joins, predicates, tables, samples, label, template_ids, num_queries_per_temp, shuffle_id_all_queries

def load_data_detect(file_name, num_materialized_samples, num_queries_per_file, seed=0):
    np.random.seed(seed)
    random.seed(seed)

    joins = []
    predicates = []
    tables = []
    samples = []
    label = []
    numerical_label = []

    templates = {}
    template_joins = {}
    template_ids = []
    temp_counts = []

    # Load queries
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

    # Load bitmaps
    num_bytes_per_bitmap = int((num_materialized_samples + 7) >> 3)
    with open(file_name + ".bitmaps", 'rb') as f:
        for i in range(len(tables)):
            four_bytes = f.read(4)
            if not four_bytes:
                print("Error while reading 'four_bytes'")
                exit(1)
            num_bitmaps_curr_query = int.from_bytes(four_bytes, byteorder='little')
            bitmaps = np.empty((num_bitmaps_curr_query, num_bytes_per_bitmap * 8), dtype=np.uint8)
            for j in range(num_bitmaps_curr_query):
                # Read bitmap
                bitmap_bytes = f.read(num_bytes_per_bitmap)
                if not bitmap_bytes:
                    print("Error while reading 'bitmap_bytes'")
                    exit(1)
                bitmaps[j] = np.unpackbits(np.frombuffer(bitmap_bytes, dtype=np.uint8))
            samples.append(bitmaps)
    print("Loaded bitmaps")

    # Split predicates
    predicates = [list(chunks(d, 3)) for d in predicates]

    filtered_temp_list_all = [i for i in range(len(temp_counts)) if temp_counts[i] >= 500]


    query_id_list = []

    for temp_id in filtered_temp_list_all:
        total_id_per_tmp = [i for i in range(num_queries_per_file) if template_ids[i] == temp_id]
        shuffle(total_id_per_tmp)
        query_id_list.append(total_id_per_tmp)

    joins_list = []
    predicates_list = []
    tables_list = []
    samples_list = []
    label_list = []
    template_ids_list = []
    num_queries_per_tmp = []

    for shuffle_id_all_queries in query_id_list:
        joins_list.extend([joins[x] for x in shuffle_id_all_queries])
        predicates_list.extend([predicates[x] for x in shuffle_id_all_queries])
        tables_list.extend([tables[x] for x in shuffle_id_all_queries])
        samples_list.extend([samples[x] for x in shuffle_id_all_queries])
        label_list.extend([label[x] for x in shuffle_id_all_queries])
        template_ids_list.extend([template_ids[x] for x in shuffle_id_all_queries])
        num_queries_per_tmp.append(len(shuffle_id_all_queries))

    return joins_list, predicates_list, tables_list, samples_list, label_list, template_ids_list, num_queries_per_tmp

def load_and_encode_train_data_both_burnin(num_materialized_samples, is_imbalance=False, num_queries_train_per_task=3600, num_queries_test=400, num_tasks=3, num_burnin=4, seed=0):
    # num_tasks:  number of templates!!

    file_name_queries = "data/train"
    file_name_column_min_max_vals = "data/column_min_max_vals.csv"

    np.random.seed(seed)
    random.seed(seed)

    num_queries_per_file = 100000
    # num_queries_per_temp = num_queries_train_per_task + num_queries_test
    joins, predicates, tables, samples, label, template_ids, num_queries_list, shuffle_id_all_queries = load_data_both(file_name_queries,
                                                                                num_materialized_samples, num_queries_per_file, num_queries_per_task=num_queries_train_per_task + num_queries_test, num_task=num_tasks,seed=seed)

    # Get column name dict
    column_names = get_all_column_names(predicates)
    column2vec, idx2column = get_set_encoding(column_names)

    # Get table name dict
    table_names = get_all_table_names(tables)
    table2vec, idx2table = get_set_encoding(table_names)

    # Get operator name dict
    operators = get_all_operators(predicates)
    op2vec, idx2op = get_set_encoding(operators)

    # Get join name dict
    join_set = get_all_joins(joins)
    join2vec, idx2join = get_set_encoding(join_set)

    # Get min and max values for each column
    with open(file_name_column_min_max_vals, 'rU') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter=','))
        column_min_max_vals = {}
        for i, row in enumerate(data_raw):
            if i == 0:
                continue
            column_min_max_vals[row[0]] = [float(row[1]), float(row[2])]

    # Get feature encoding and proper normalization
    samples_enc = encode_samples(tables, samples, table2vec)
    predicates_enc, joins_enc = encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec)
    label_norm, min_val, max_val = normalize_labels(label)

    # Split in training and validation samples

    if is_imbalance:
        ratio_per_temp = [5, 20, 75]
    else:
        ratio_per_temp = [33, 33, 34]

    ratio_per_temp.reverse()

    samples_train = []
    predicates_train = []
    joins_train = []
    labels_train = []
    template_ids_train = []

    samples_test = []
    predicates_test = []
    joins_test = []
    labels_test = []
    template_ids_test = []

    for _ in range(num_burnin):
        for temp_id in range(len(ratio_per_temp)):
            num_candi_queries_train_per_temp = num_queries_list[temp_id] - num_queries_test

            num_q_so_far = sum(num_queries_list[:temp_id])

            candi_samples_enc = samples_enc[num_q_so_far:num_q_so_far + num_candi_queries_train_per_temp] # leave out the test set!
            candi_predicates_enc = predicates_enc[num_q_so_far:num_q_so_far +num_candi_queries_train_per_temp]
            candi_joins_enc = joins_enc[num_q_so_far:num_q_so_far +num_candi_queries_train_per_temp]
            candi_label_norm = label_norm[num_q_so_far:num_q_so_far +num_candi_queries_train_per_temp]
            candi_template_ids = template_ids[num_q_so_far:num_q_so_far +num_candi_queries_train_per_temp]

            temp = list(zip(candi_samples_enc, candi_predicates_enc, candi_joins_enc, candi_label_norm, candi_template_ids))
            random.shuffle(temp)
            candi_samples_enc, candi_predicates_enc, candi_joins_enc, candi_label_norm, candi_template_ids = zip(*temp)
            candi_samples_enc, candi_predicates_enc, candi_joins_enc, candi_label_norm, candi_template_ids= list(candi_samples_enc), \
                                                                            list(candi_predicates_enc), list(candi_joins_enc), list(candi_label_norm), list(candi_template_ids)

            num_queries_train_per_temp = int(ratio_per_temp[temp_id] * num_queries_train_per_task / sum(ratio_per_temp))

            samples_train.extend(candi_samples_enc[:num_queries_train_per_temp])
            predicates_train.extend(candi_predicates_enc[:num_queries_train_per_temp])
            joins_train.extend(candi_joins_enc[:num_queries_train_per_temp])
            labels_train.extend(candi_label_norm[:num_queries_train_per_temp])
            template_ids_train.extend(candi_template_ids[:num_queries_train_per_temp])

            if len(candi_samples_enc) < num_queries_train_per_temp:
                sampled_ids = random.choices(range(len(candi_samples_enc)), k=(num_queries_train_per_temp-len(candi_samples_enc)))
                for sampled_id in sampled_ids:
                    samples_train.append(candi_samples_enc[sampled_id])
                    predicates_train.append(candi_predicates_enc[sampled_id])
                    joins_train.append(candi_joins_enc[sampled_id])
                    labels_train.append(candi_label_norm[sampled_id])
                    template_ids_train.append(candi_template_ids[sampled_id])

    for temp_id in range(len(ratio_per_temp)):
        num_candi_queries_train_per_temp = num_queries_list[temp_id] - num_queries_test

        num_q_so_far = sum(num_queries_list[:temp_id])
        candi_samples_enc = samples_enc[num_q_so_far:num_q_so_far + num_candi_queries_train_per_temp]  # leave out the test set!
        candi_predicates_enc = predicates_enc[num_q_so_far:num_q_so_far + num_candi_queries_train_per_temp]
        candi_joins_enc = joins_enc[num_q_so_far:num_q_so_far + num_candi_queries_train_per_temp]
        candi_label_norm = label_norm[num_q_so_far:num_q_so_far + num_candi_queries_train_per_temp]
        candi_template_ids = template_ids[num_q_so_far:num_q_so_far + num_candi_queries_train_per_temp]

        temp = list(zip(candi_samples_enc, candi_predicates_enc, candi_joins_enc, candi_label_norm, candi_template_ids))
        random.shuffle(temp)
        candi_samples_enc, candi_predicates_enc, candi_joins_enc, candi_label_norm, candi_template_ids = zip(*temp)
        candi_samples_enc, candi_predicates_enc, candi_joins_enc, candi_label_norm, candi_template_ids = list(
            candi_samples_enc), list(candi_predicates_enc), list(candi_joins_enc), list(candi_label_norm), list(candi_template_ids)
        num_queries_train_per_temp = num_queries_train_per_task

        samples_train.extend(candi_samples_enc[:num_queries_train_per_temp])
        predicates_train.extend(candi_predicates_enc[:num_queries_train_per_temp])
        joins_train.extend(candi_joins_enc[:num_queries_train_per_temp])
        labels_train.extend(candi_label_norm[:num_queries_train_per_temp])
        template_ids_train.extend(candi_template_ids[:num_queries_train_per_temp])

        if len(candi_samples_enc) < num_queries_train_per_temp:
            sampled_ids = random.choices(range(len(candi_samples_enc)),
                                         k=(num_queries_train_per_temp - len(candi_samples_enc)))
            for sampled_id in sampled_ids:
                samples_train.append(candi_samples_enc[sampled_id])
                predicates_train.append(candi_predicates_enc[sampled_id])
                joins_train.append(candi_joins_enc[sampled_id])
                labels_train.append(candi_label_norm[sampled_id])
                template_ids_train.append(candi_template_ids[sampled_id])

    test_id_list = []
    for temp_id in range(len(ratio_per_temp)):
        num_q_so_far = sum(num_queries_list[:temp_id]) + num_queries_list[temp_id] - num_queries_test

        samples_test.extend(samples_enc[num_q_so_far: num_q_so_far+num_queries_test])
        predicates_test.extend(predicates_enc[num_q_so_far: num_q_so_far+num_queries_test])
        joins_test.extend(joins_enc[num_q_so_far: num_q_so_far+num_queries_test])
        labels_test.extend(label_norm[num_q_so_far: num_q_so_far+num_queries_test])
        template_ids_test.extend(template_ids[num_q_so_far: num_q_so_far+num_queries_test])
        test_id_list.append(shuffle_id_all_queries[num_q_so_far: num_q_so_far+num_queries_test])

    print("Number of training samples: {}".format(len(labels_train)))
    print("Number of validation samples: {}".format(len(labels_test)))

    max_num_joins = max(max([len(j) for j in joins_train]), max([len(j) for j in joins_test]))
    max_num_predicates = max(max([len(p) for p in predicates_train]), max([len(p) for p in predicates_test]))

    dicts = [table2vec, column2vec, op2vec, join2vec]
    train_data = [samples_train, predicates_train, joins_train]
    test_data = [samples_test, predicates_test, joins_test]

    return dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, template_ids_train, template_ids_test,\
           max_num_joins, max_num_predicates, train_data, test_data, test_id_list

def load_and_encode_train_data_detect(num_materialized_samples, seed=0):
    # num_tasks:  number of templates!!

    file_name_queries = "data/train"
    file_name_column_min_max_vals = "data/column_min_max_vals.csv"

    np.random.seed(seed)
    random.seed(seed)

    num_queries_per_file = 100000
    # num_queries_per_temp = num_queries_train_per_task + num_queries_test
    joins, predicates, tables, samples, label, template_ids, num_queries_per_tmp = load_data_detect(file_name_queries,
                                                                                num_materialized_samples, num_queries_per_file, seed=seed)

    num_tmp = len(num_queries_per_tmp)
    # Get column name dict
    column_names = get_all_column_names(predicates)
    column2vec, idx2column = get_set_encoding(column_names)

    # Get table name dict
    table_names = get_all_table_names(tables)
    table2vec, idx2table = get_set_encoding(table_names)

    # Get operator name dict
    operators = get_all_operators(predicates)
    op2vec, idx2op = get_set_encoding(operators)

    # Get join name dict
    join_set = get_all_joins(joins)
    join2vec, idx2join = get_set_encoding(join_set)

    # Get min and max values for each column
    with open(file_name_column_min_max_vals, 'rU') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter=','))
        column_min_max_vals = {}
        for i, row in enumerate(data_raw):
            if i == 0:
                continue
            column_min_max_vals[row[0]] = [float(row[1]), float(row[2])]

    # Get feature encoding and proper normalization
    samples_enc = encode_samples(tables, samples, table2vec)
    predicates_enc, joins_enc = encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec)
    label_norm, min_val, max_val = normalize_labels(label)

    num_pairs = 1000
    num_batch = 100

    samples_diff = []
    predicates_diff = []
    joins_diff = []
    labels_diff = []
    template_ids_diff = []

    samples_same = []
    predicates_same = []
    joins_same = []
    labels_same = []
    template_ids_same = []

    for _ in range(num_pairs):
        res_a_pair = []
        tmp_ids = random.sample(range(num_tmp), 2)
        for tmp_id in tmp_ids:
            num_q_so_far = sum(num_queries_per_tmp[:tmp_id])
            candi_samples_enc = samples_enc[num_q_so_far:num_q_so_far + num_queries_per_tmp[tmp_id]]
            candi_predicates_enc = predicates_enc[num_q_so_far:num_q_so_far + num_queries_per_tmp[tmp_id]]
            candi_joins_enc = joins_enc[num_q_so_far:num_q_so_far + num_queries_per_tmp[tmp_id]]
            candi_label_norm = label_norm[num_q_so_far:num_q_so_far + num_queries_per_tmp[tmp_id]]
            candi_template_ids = template_ids[num_q_so_far:num_q_so_far + num_queries_per_tmp[tmp_id]]


            temp = list(
                zip(candi_samples_enc, candi_predicates_enc, candi_joins_enc, candi_label_norm, candi_template_ids))
            random.shuffle(temp)
            candi_samples_enc, candi_predicates_enc, candi_joins_enc, candi_label_norm, candi_template_ids = zip(*temp)
            candi_samples_enc, candi_predicates_enc, candi_joins_enc, candi_label_norm, candi_template_ids = list(
                candi_samples_enc), list(candi_predicates_enc), list(candi_joins_enc), list(candi_label_norm), list(
                candi_template_ids)

            samples_diff.extend(candi_samples_enc[:num_batch])
            predicates_diff.extend(candi_predicates_enc[:num_batch])
            joins_diff.extend(candi_joins_enc[:num_batch])
            labels_diff.extend(candi_label_norm[:num_batch])
            template_ids_diff.extend(candi_template_ids[:num_batch])


    for _ in range(num_pairs):
        tmp_id = random.sample(range(num_tmp), 1)[0]

        for _ in range(2):
            num_q_so_far = sum(num_queries_per_tmp[:tmp_id])
            candi_samples_enc = samples_enc[num_q_so_far:num_q_so_far + num_queries_per_tmp[tmp_id]]
            candi_predicates_enc = predicates_enc[num_q_so_far:num_q_so_far + num_queries_per_tmp[tmp_id]]
            candi_joins_enc = joins_enc[num_q_so_far:num_q_so_far + num_queries_per_tmp[tmp_id]]
            candi_label_norm = label_norm[num_q_so_far:num_q_so_far + num_queries_per_tmp[tmp_id]]
            candi_template_ids = template_ids[num_q_so_far:num_q_so_far + num_queries_per_tmp[tmp_id]]

            temp = list(
                zip(candi_samples_enc, candi_predicates_enc, candi_joins_enc, candi_label_norm, candi_template_ids))
            random.shuffle(temp)
            candi_samples_enc, candi_predicates_enc, candi_joins_enc, candi_label_norm, candi_template_ids = zip(*temp)
            candi_samples_enc, candi_predicates_enc, candi_joins_enc, candi_label_norm, candi_template_ids = list(
                candi_samples_enc), list(candi_predicates_enc), list(candi_joins_enc), list(candi_label_norm), list(
                candi_template_ids)

            samples_same.extend(candi_samples_enc[:num_batch])
            predicates_same.extend(candi_predicates_enc[:num_batch])
            joins_same.extend(candi_joins_enc[:num_batch])
            labels_same.extend(candi_label_norm[:num_batch])
            template_ids_same.extend(candi_template_ids[:num_batch])

    max_num_joins = max(max([len(j) for j in joins_diff]), max([len(j) for j in joins_same]))
    max_num_predicates = max(max([len(p) for p in predicates_diff]), max([len(p) for p in predicates_same]))

    dicts = [table2vec, column2vec, op2vec, join2vec]
    diff_data = [samples_diff, predicates_diff, joins_diff]
    same_data = [samples_same, predicates_same, joins_same]


    return dicts, column_min_max_vals, min_val, max_val, labels_diff, labels_same,\
           max_num_joins, max_num_predicates, diff_data, same_data

def make_dataset(samples, predicates, joins, labels, max_num_joins, max_num_predicates):
    """Add zero-padding and wrap as tensor dataset."""

    sample_masks = []
    sample_tensors = []
    for sample in samples:
        sample_tensor = np.vstack(sample)
        num_pad = max_num_joins + 1 - sample_tensor.shape[0]
        sample_mask = np.ones_like(sample_tensor).mean(1, keepdims=True)
        sample_tensor = np.pad(sample_tensor, ((0, num_pad), (0, 0)), 'constant')
        sample_mask = np.pad(sample_mask, ((0, num_pad), (0, 0)), 'constant')
        sample_tensors.append(np.expand_dims(sample_tensor, 0))
        sample_masks.append(np.expand_dims(sample_mask, 0))
    sample_tensors = np.vstack(sample_tensors)
    sample_tensors = torch.FloatTensor(sample_tensors)
    sample_masks = np.vstack(sample_masks)
    sample_masks = torch.FloatTensor(sample_masks)

    predicate_masks = []
    predicate_tensors = []
    for predicate in predicates:
        predicate_tensor = np.vstack(predicate)
        num_pad = max_num_predicates - predicate_tensor.shape[0]
        predicate_mask = np.ones_like(predicate_tensor).mean(1, keepdims=True)
        predicate_tensor = np.pad(predicate_tensor, ((0, num_pad), (0, 0)), 'constant')
        predicate_mask = np.pad(predicate_mask, ((0, num_pad), (0, 0)), 'constant')
        predicate_tensors.append(np.expand_dims(predicate_tensor, 0))
        predicate_masks.append(np.expand_dims(predicate_mask, 0))
    predicate_tensors = np.vstack(predicate_tensors)
    predicate_tensors = torch.FloatTensor(predicate_tensors)
    predicate_masks = np.vstack(predicate_masks)
    predicate_masks = torch.FloatTensor(predicate_masks)

    join_masks = []
    join_tensors = []
    for join in joins:
        join_tensor = np.vstack(join)
        num_pad = max_num_joins - join_tensor.shape[0]
        join_mask = np.ones_like(join_tensor).mean(1, keepdims=True)
        join_tensor = np.pad(join_tensor, ((0, num_pad), (0, 0)), 'constant')
        join_mask = np.pad(join_mask, ((0, num_pad), (0, 0)), 'constant')
        join_tensors.append(np.expand_dims(join_tensor, 0))
        join_masks.append(np.expand_dims(join_mask, 0))
    join_tensors = np.vstack(join_tensors)
    join_tensors = torch.FloatTensor(join_tensors)
    join_masks = np.vstack(join_masks)
    join_masks = torch.FloatTensor(join_masks)

    target_tensor = torch.FloatTensor(labels)
    return dataset.TensorDataset(sample_tensors, predicate_tensors, join_tensors, target_tensor, sample_masks,
                                 predicate_masks, join_masks)

def get_workloads(num_queries, num_materialized_samples,is_imbalance=False, num_train=1000, num_test=100, num_tasks=3, num_burnin=4, seed=0):
    dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, template_ids_train, template_ids_test, \
    max_num_joins, max_num_predicates, train_data, test_data, test_id_list = load_and_encode_train_data_both_burnin(num_materialized_samples,is_imbalance=is_imbalance,num_tasks=num_tasks,
                                                                                                      num_queries_train_per_task=num_train, num_queries_test=num_test,
                                                                                                      num_burnin=num_burnin,seed=seed)
    train_dataset = make_dataset(*train_data, labels=labels_train, max_num_joins=max_num_joins,
                                 max_num_predicates=max_num_predicates)
    print("Created TensorDataset for training data")
    test_dataset = make_dataset(*test_data, labels=labels_test, max_num_joins=max_num_joins,
                                max_num_predicates=max_num_predicates)
    print("Created TensorDataset for validation data")
    return dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_predicates, train_dataset, test_dataset, test_id_list

def get_dataset_detect(num_materialized_samples, seed=0):
    dicts, column_min_max_vals, min_val, max_val, labels_diff, labels_same, \
    max_num_joins, max_num_predicates, diff_data, same_data = load_and_encode_train_data_detect(num_materialized_samples,seed=seed)
    diff_dataset = make_dataset(*diff_data, labels=labels_diff, max_num_joins=max_num_joins,
                                 max_num_predicates=max_num_predicates)
    print("Created TensorDataset for training data")
    same_dataset = make_dataset(*same_data, labels=labels_same, max_num_joins=max_num_joins,
                                max_num_predicates=max_num_predicates)
    print("Created TensorDataset for validation data")
    return dicts, column_min_max_vals, min_val, max_val, labels_diff, labels_same,\
           max_num_joins, max_num_predicates, diff_dataset, same_dataset

def get_task(task_id, batch_size, queries_train, queries_test, num_queries_per_workload=900, num_test_queries_per_workload=100, seed=0):
    """
    Returns a single task of a split workload
    :param task_id:
    :param batch_size:
    :return:
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    start_id = (task_id)* num_queries_per_workload
    end_id = (task_id+1) * num_queries_per_workload

    test_start_id = (task_id) * num_test_queries_per_workload
    test_end_id = (task_id+1) * num_test_queries_per_workload
    
    # print("task_id {}".format(task_id))
    # print("num_queries_per_workload {}".format(num_queries_per_workload))
    # print("num_test_queries_per_workload {}".format(num_test_queries_per_workload))

    train_loader = torch.utils.data.DataLoader(torch.utils.data.dataset.Subset(queries_train, list(range(start_id,end_id))), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.dataset.Subset(queries_test, list(range(test_start_id, test_end_id))), batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

def get_task_test_queries(task_id, batch_size, queries_test, num_test_queries_per_workload=100, seed=0):
    # task_id: the id current task
    start_id = (task_id) * num_test_queries_per_workload
    end_id = (task_id + 1) * num_test_queries_per_workload

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.dataset.Subset(queries_test, list(range(start_id, end_id))), batch_size=batch_size)

    return test_loader