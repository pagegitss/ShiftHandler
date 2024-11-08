import json
import numpy as np
import torch
import torch.optim
import joblib
import os
import math
from sklearn import preprocessing
from sklearn.pipeline import Pipeline

from torch.utils.data import DataLoader
import net
from featurize import TreeFeaturizer
from TreeConvolution.util import prepare_trees

CUDA = torch.cuda.is_available()

def _nn_path(base):
    return os.path.join(base, "nn_weights")

def _x_transform_path(base):
    return os.path.join(base, "x_transform")

def _y_transform_path(base):
    return os.path.join(base, "y_transform")

def _channels_path(base):
    return os.path.join(base, "channels")

def _n_path(base):
    return os.path.join(base, "n")


def _inv_log1p(x):
    return np.exp(x) - 1

class BaoData:
    def __init__(self, data):
        assert data
        self.__data = data

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, idx):
        return (self.__data[idx]["tree"],
                self.__data[idx]["target"])

def collate(x):
    trees = []
    targets = []

    for tree, target in x:
        trees.append(tree)
        targets.append(target)

    targets = np.array(targets)
    targets = torch.tensor(targets)
    return trees, targets

def left_child(x):
    if len(x) != 3:
        return None
    return x[1]

def right_child(x):
    if len(x) != 3:
        return None
    return x[2]

def features(x):
    return x[0]

class BaoRegression:
    def __init__(self, verbose=False, have_cache_data=False):
        self.__net = None
        self.__verbose = verbose

        log_transformer = preprocessing.FunctionTransformer(
            np.log1p, _inv_log1p,
            validate=True)
        scale_transformer = preprocessing.MinMaxScaler()

        self.__pipeline = Pipeline([("log", log_transformer),
                                    ("scale", scale_transformer)])
        
        self.__tree_transform = TreeFeaturizer()
        self.__have_cache_data = have_cache_data
        self.__in_channels = None
        self.__n = 0
        
    def __log(self, *args):
        if self.__verbose:
            print(*args)

    def num_items_trained_on(self):
        return self.__n
            
    def load(self, path):
        with open(_n_path(path), "rb") as f:
            self.__n = joblib.load(f)
        with open(_channels_path(path), "rb") as f:
            self.__in_channels = joblib.load(f)
            
        self.__net = net.BaoNet(self.__in_channels)
        self.__net.load_state_dict(torch.load(_nn_path(path)))
        self.__net.eval()


        with open(_y_transform_path(path), "rb") as f:
            self.__pipeline = joblib.load(f)
        with open(_x_transform_path(path), "rb") as f:
            self.__tree_transform = joblib.load(f)

    def save(self, path):
        # try to create a directory here
        os.makedirs(path, exist_ok=True)
        
        torch.save(self.__net.state_dict(), _nn_path(path))
        with open(_y_transform_path(path), "wb") as f:
            joblib.dump(self.__pipeline, f)
        # can query feature extractor be pretrained and fixed during runtime?
        with open(_x_transform_path(path), "wb") as f:
            joblib.dump(self.__tree_transform, f)
        with open(_channels_path(path), "wb") as f:
            joblib.dump(self.__in_channels, f)
        with open(_n_path(path), "wb") as f:
            joblib.dump(self.__n, f)

    def save_feature_extractor(self, path):
        # try to create a directory here
        os.makedirs(path, exist_ok=True)

        with open(_y_transform_path(path), "wb") as f:
            joblib.dump(self.__pipeline, f)
        # can query feature extractor be pretrained and fixed during runtime?
        with open(_x_transform_path(path), "wb") as f:
            joblib.dump(self.__tree_transform, f)
        with open(_channels_path(path), "wb") as f:
            joblib.dump(self.__in_channels, f)
        with open(_n_path(path), "wb") as f:
            joblib.dump(self.__n, f)

    def fit_feature_extractor(self, X, y):
        if isinstance(y, list):
            y = np.array(y)

        X = [json.loads(x) if isinstance(x, str) else x for x in X]
        self.__n = len(X)

        # transform the set of trees into feature vectors using a log
        # (assuming the tail behavior exists, TODO investigate
        #  the quantile transformer from scikit)
        y = self.__pipeline.fit_transform(y.reshape(-1, 1)).astype(np.float32)

        self.__tree_transform.fit(X)
        X = self.__tree_transform.transform(X)

        pairs = list(zip(X, y))
        dataset = DataLoader(pairs,
                             batch_size=16,
                             shuffle=True,
                             collate_fn=collate)


        # determine the initial number of channels
        for inp, _tar in dataset:
            in_channels = inp[0][0].shape[0]
            break

        self.__log("Initial input channels:", in_channels)

        if self.__have_cache_data:
            assert in_channels == self.__tree_transform.num_operators() + 3
        else:
            assert in_channels == self.__tree_transform.num_operators() + 2

        self.__net = net.BaoNet(in_channels)
        self.__in_channels = in_channels
        if CUDA:
            self.__net = self.__net.cuda()

    def fit_model(self, X, y, tradeoff=0.5, size_current_batch=None, ada_size=False, seed=0):
        if isinstance(y, list):
            y = np.array(y)

        torch.manual_seed(seed)

        current_losses_list = []
        idx_list = []
        replay_losses_list = []
        replay_idx_list = []

        self.__net.train()
        X = [json.loads(x) if isinstance(x, str) else x for x in X]

        # transform the set of trees into feature vectors using a log
        # (assuming the tail behavior exists, TODO investigate
        #  the quantile transformer from scikit)
        y = self.__pipeline.transform(y.reshape(-1, 1)).astype(np.float32)
        X = self.__tree_transform.transform(X)

        if size_current_batch is None:
            pairs = list(zip(X, y))
            dataset_shuffle_list = torch.randperm(len(pairs)).tolist()
            shuffled_dataset = torch.utils.data.Subset(pairs, dataset_shuffle_list)
            dataset = DataLoader(shuffled_dataset,
                                 batch_size=16,
                                 shuffle=False,
                                 collate_fn=collate)
            index_replay = DataLoader(dataset_shuffle_list,
                                      batch_size=16,
                                      shuffle=False)
        else:
            pairs_current = list(zip(X[:size_current_batch], y[:size_current_batch]))
            pairs_replay = list(zip(X[size_current_batch:], y[size_current_batch:]))
            current_bs = 16
            # replay_bs = 5
            replay_bs = math.ceil(len(pairs_replay) / math.ceil(len(pairs_current) / current_bs))
            dataset_shuffle_list = torch.randperm(len(pairs_current)).tolist()
            shuffled_dataset = torch.utils.data.Subset(pairs_current, dataset_shuffle_list)
            dataset = DataLoader(shuffled_dataset,
                                 batch_size=current_bs,
                                 shuffle=False,
                                 collate_fn=collate)
            index_replay = DataLoader(dataset_shuffle_list,
                                      batch_size=current_bs,
                                      shuffle=False)
            if not ada_size:
                dataset_replay = DataLoader(pairs_replay,
                                            batch_size=replay_bs,
                                            shuffle=True,
                                            collate_fn=collate)

            else:
                replay_shuffle_list = torch.randperm(len(pairs_replay)).tolist()
                shuffled_dataset_replay = torch.utils.data.Subset(pairs_replay, replay_shuffle_list)
                dataset_replay = DataLoader(shuffled_dataset_replay,
                                            batch_size=replay_bs,
                                            shuffle=False,
                                            collate_fn=collate)
                replay_index_replay = DataLoader(replay_shuffle_list,
                                                 batch_size=replay_bs,
                                                 shuffle=False)
        # determine the initial number of channels
        for inp, _tar in dataset:
            in_channels = inp[0][0].shape[0]
            break

        self.__log("Initial input channels:", in_channels)

        if self.__have_cache_data:
            assert in_channels == self.__tree_transform.num_operators() + 3
        else:
            assert in_channels == self.__tree_transform.num_operators() + 2

        self.__net = net.BaoNet(in_channels)
        self.__in_channels = in_channels
        if CUDA:
            self.__net = self.__net.cuda()

        optimizer = torch.optim.Adam(self.__net.parameters())
        loss_fn = torch.nn.MSELoss(reduction='none')

        losses = []
        for epoch in range(100):
            loss_accum = 0

            if size_current_batch is None:
                if ada_size:
                    current_losses_list = []
                    idx_list = []
                    replay_losses_list = []
                    replay_idx_list = []
                index_iter = iter(index_replay)
                for x, y in dataset:
                    idxs = next(index_iter)
                    idx_list.extend(idxs)
                    if CUDA:
                        y = y.cuda()
                    y_pred = self.__net(x)
                    current_losses = loss_fn(y_pred, y)
                    loss = torch.mean(current_losses)
                    current_losses_list.extend(list(current_losses.cpu().detach().numpy()))

                    loss_accum += loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                loss_accum /= len(dataset)
            else:
                if ada_size:
                    current_losses_list = []
                    idx_list = []
                    replay_losses_list = []
                    replay_idx_list = []
                dataset_replay_iterator = iter(dataset_replay)
                if ada_size:
                    index_iter = iter(index_replay)
                    index_replay_iter = iter(replay_index_replay)
                for (x, y) in dataset:
                    if ada_size:
                        idxs = next(index_iter)
                    try:
                        (x_r, y_r) = next(dataset_replay_iterator)
                        if ada_size:
                            replay_idxs = next(index_replay_iter)
                    except StopIteration:
                        dataset_replay_iterator = iter(dataset_replay)
                        (x_r, y_r) = next(dataset_replay_iterator)
                        if ada_size:
                            index_replay_iter = iter(replay_index_replay)
                            replay_idxs = next(index_replay_iter)

                    if CUDA:
                        y = y.cuda()
                        y_r = y_r.cuda()
                    y_pred = self.__net(x)
                    y_r_pred = self.__net(x_r)
                    current_losses = loss_fn(y_pred, y)
                    replay_losses = loss_fn(y_r_pred, y_r)

                    current_loss_mean = torch.mean(current_losses)
                    replay_loss_mean = torch.mean(replay_losses)
                    loss = tradeoff * current_loss_mean + (1 - tradeoff) * replay_loss_mean

                    replay_losses_a_iter = list(replay_losses.cpu().detach().numpy())

                    if ada_size:
                        replay_losses_list.extend(replay_losses_a_iter)
                        replay_idx_list.extend(replay_idxs)
                        current_losses_list.extend(list(current_losses.cpu().detach().numpy()))
                        idx_list.extend(idxs)

                    loss_accum += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                loss_accum /= len(dataset) + len(dataset_replay)

            losses.append(loss_accum)
            if epoch % 15 == 0:
                self.__log("Epoch", epoch, "training loss:", loss_accum)

            # stopping condition
            if len(losses) > 10 and losses[-1] < 0.1:
                last_two = np.min(losses[-2:])
                if last_two > losses[-10] or (losses[-10] - last_two < 0.0001):
                    self.__log("Stopped training from convergence condition at epoch", epoch)
                    break
        else:
            self.__log("Stopped training after max epochs")

        return idx_list, current_losses_list, replay_idx_list, replay_losses_list

    def fit(self, X, y, tradeoff=0.5, size_current_batch=None):
        if isinstance(y, list):
            y = np.array(y)
        self.__net.train()
        X = [json.loads(x) if isinstance(x, str) else x for x in X]
        self.__n = len(X)
            
        # transform the set of trees into feature vectors using a log
        # (assuming the tail behavior exists, TODO investigate
        #  the quantile transformer from scikit)
        y = self.__pipeline.fit_transform(y.reshape(-1, 1)).astype(np.float32)

        self.__tree_transform.fit(X)
        X = self.__tree_transform.transform(X)

        if size_current_batch is None:
            pairs = list(zip(X, y))
            dataset = DataLoader(pairs,
                                 batch_size=16,
                                 shuffle=True,
                                 collate_fn=collate)
        else:
            pairs_current = list(zip(X[:size_current_batch], y[:size_current_batch]))
            pairs_replay = list(zip(X[size_current_batch:], y[size_current_batch:]))
            current_bs = 16
            replay_bs = 5
            dataset = DataLoader(pairs_current,
                                 batch_size=current_bs,
                                 shuffle=True,
                                 collate_fn=collate)
            dataset_replay = DataLoader(pairs_replay,
                                 batch_size=replay_bs,
                                 shuffle=True,
                                 collate_fn=collate)


        # determine the initial number of channels
        for inp, _tar in dataset:
            in_channels = inp[0][0].shape[0]
            break

        self.__log("Initial input channels:", in_channels)

        if self.__have_cache_data:
            assert in_channels == self.__tree_transform.num_operators() + 3
        else:
            assert in_channels == self.__tree_transform.num_operators() + 2

        self.__net = net.BaoNet(in_channels)
        self.__in_channels = in_channels
        if CUDA:
            self.__net = self.__net.cuda()

        optimizer = torch.optim.Adam(self.__net.parameters())
        loss_fn = torch.nn.MSELoss()
        
        losses = []
        for epoch in range(100):
            loss_accum = 0

            if size_current_batch is None:
                for x, y in dataset:
                    if CUDA:
                        y = y.cuda()
                    y_pred = self.__net(x)
                    loss = loss_fn(y_pred, y)
                    loss_accum += loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                loss_accum /= len(dataset)
            else:
                dataset_replay_iterator = iter(dataset_replay)
                for (x, y) in dataset:
                    try:
                        (x_r, y_r) = next(dataset_replay_iterator)
                    except StopIteration:
                        dataset_replay_iterator = iter(dataset_replay)
                        (x_r, y_r) = next(dataset_replay_iterator)

                    if CUDA:
                        y = y.cuda()
                        y_r = y_r.cuda()
                    y_pred = self.__net(x)
                    loss = tradeoff*loss_fn(y_pred, y)

                    y_r_pred = self.__net(x_r)
                    loss += (1-tradeoff) * loss_fn(y_r_pred, y_r)

                    loss_accum += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                loss_accum /= len(dataset) + len(dataset_replay)

            losses.append(loss_accum)
            if epoch % 15 == 0:
                self.__log("Epoch", epoch, "training loss:", loss_accum)

            # stopping condition
            if len(losses) > 10 and losses[-1] < 0.1:
                last_two = np.min(losses[-2:])
                if last_two > losses[-10] or (losses[-10] - last_two < 0.0001):
                    self.__log("Stopped training from convergence condition at epoch", epoch)
                    break
        else:
            self.__log("Stopped training after max epochs")

    def get_query_features(self, X):
        X = [json.loads(x) if isinstance(x, str) else x for x in X]
        self.__n = len(X)
        X = self.__tree_transform.transform(X)
        with torch.no_grad():
            fixed_features = self.__net.get_fixed_features(X)

        return fixed_features

    def get_before_features(self, X):
        X = [json.loads(x) if isinstance(x, str) else x for x in X]
        self.__n = len(X)
        X = self.__tree_transform.transform(X)
        fixed_features = self.__net.get_before_features(X)

        return fixed_features

    def predict(self, X):
        if not isinstance(X, list):
            X = [X]
        X = [json.loads(x) if isinstance(x, str) else x for x in X]

        X = self.__tree_transform.transform(X)
        
        self.__net.eval()
        pred = self.__net(X).cpu().detach().numpy()
        return self.__pipeline.inverse_transform(pred)

