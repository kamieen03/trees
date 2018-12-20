import numpy as np
import random
import copy

class RegressionTree:
    def __init__(self, parent=None, split_f_v=None, part_of_forest = False):
        self.parent = parent
        self.children = []
        self.label = None
        self.splitFunctionValue = split_f_v
        self.splitFunction = lambda x: True
        self.feature_threshold = None
        self.split_features_num = None
        self.left_ends = []
        self.right_ends = []
        self.widths = []
        self.PART_OF_FOREST = part_of_forest
        if self.parent is not None:
            self.split_features_num = self.parent.split_features_num
            self.PART_OF_FOREST = self.parent.PART_OF_FOREST

            self.left_ends = copy.copy(self.parent.left_ends)
            self.right_ends = copy.copy(self.parent.right_ends)
            self.widths = copy.copy(self.parent.widths)
            index, t = self.parent.feature_threshold
            if self.splitFunctionValue:
                self.right_ends[index] = t
            else:
                self.left_ends[index] = t
            self.widths[index] = self.right_ends[index] - self.left_ends[index]

    def split(self, data):
        ys = data[:, 1]
        if homogeneous(ys) or any([w < 0.001 for w in self.widths]):
            return majorityVote(data, self)

        indices = self.generate_indices()
        index_threshs = self.generate_splits(indices)
        if index_threshs == []:
            return majorityVote(data, self)
        best_it = max(index_threshs, key=lambda it_pair: gain(data, it_pair))

        if gain(data, best_it) < 0.001:
            return majorityVote(data, self)

        self.splitFunction = lambda x: x[best_it[0]] < best_it[1]
        self.feature_threshold = best_it

        # add child nodes and process recursively
        for dataSubset in splitData(data, best_it):
            aChild = RegressionTree(parent=self, split_f_v=self.splitFunction(dataSubset[0][0]))
            self.children.append(aChild)
            aChild.split(dataSubset)
        return self

    def generate_splits(self, indices):
        splits = []
        for i, w in enumerate(self.widths):
            if i in indices:
                n = w ** 0.5
                if int(n) > 1:
                    for k in range(1, int(n)):
                        splits.append((i, self.left_ends[i] + k * n))
                else:
                    splits.append((i, self.left_ends[i] + w / 2))
        return splits

    def learn(self, data):
        self.split_features_num = len(data[0][0])
        mins, maxs = data[0][0].copy(), data[0][0].copy()
        for x in data[:, 0]:
            for i, val in enumerate(x):
                if val < mins[i]:
                    mins[i] = val
                elif val > maxs[i]:
                    maxs[i] = val
        widths = []
        for m, M in zip(mins, maxs):
            widths.append(M - m)
        self.left_ends = mins
        self.right_ends = maxs
        self.widths = widths
        self.split(data)

    def generate_indices(self):
        k = self.split_features_num
        if self.PART_OF_FOREST:
            k = max(1, int(k/3))
        indices = [i for i in range(self.split_features_num)]
        random.shuffle(indices)
        return indices[:k]

    def predict(self, point):
        if self.children == []:
            return self.label
        else:
            smaller = self.splitFunction(point)
            if self.children[0].splitFunctionValue == smaller:
                return self.children[0].predict(point)
            return self.children[1].predict(point)


def splitData(data, it_pair):
    index, threshold = it_pair
    for res in [True, False]:
        data_subset = np.array([row for row in data
                                if (row[0][index] < threshold) == res])
        yield data_subset


def gain(data, it_pair):
    var_gain = np.std(data[:, 1])
    for data_subset in splitData(data, it_pair):
        if data_subset.size == 0: return 0
        var_gain -= (data_subset.shape[0]) / (data.shape[0]) * np.std(data_subset[:, 1]) #ys
    return var_gain


def homogeneous(data):
    ''' Return True if the data have the same label, and False otherwise. '''
    return np.std(data) <= 0.1


def majorityVote(data, node):
    ''' Label node with the majority of the class labels in the given data set. '''
    node.label = np.mean(data[:, 1])
    return node


