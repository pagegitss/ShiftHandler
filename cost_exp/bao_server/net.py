import torch.nn as nn
import numpy
from TreeConvolution.tcnn import BinaryTreeConv, TreeLayerNorm
from TreeConvolution.tcnn import TreeActivation, DynamicPooling
from TreeConvolution.util import prepare_trees, get_fixed_trees

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

class BaoNet(nn.Module):
    def __init__(self, in_channels):
        super(BaoNet, self).__init__()
        self.__in_channels = in_channels
        self.__cuda = False

        self.fixed_feature = nn.Sequential(
            BinaryTreeConv(self.__in_channels, 256),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(256, 128),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(128, 64),
            TreeLayerNorm(),
            DynamicPooling()
        )

        self.tree_conv = nn.Sequential(
            self.fixed_feature,
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

    def in_channels(self):
        return self.__in_channels
        
    def forward(self, x):
        trees = prepare_trees(x, features, left_child, right_child,
                              cuda=self.__cuda)
        return self.tree_conv(trees)

    def get_fixed_features(self, x):
        trees = prepare_trees(x, features, left_child, right_child,
                              cuda=self.__cuda)
        return self.fixed_feature(trees)

    def get_before_features(self, x):
        trees = get_fixed_trees(x, features, left_child, right_child)
        batch_size = trees[0].shape[0]
        x1 = trees[0].reshape((batch_size, -1))
        x2 = trees[1].reshape((batch_size, -1))
        ret = numpy.hstack((x1, x2))
        return ret

    def cuda(self):
        self.__cuda = True
        return super().cuda()
