import numpy as np
import random
from AbstractTree import AbstractTree
from TestResult import TestResult
import math

class ContinuousTree(AbstractTree):
    def __init__(self, parent=None, part_of_forest = False, 
            split_features_num = None, split_f_v=None, problem = None, classes = None):
        super().__init__(parent, part_of_forest, split_features_num)

        self.problem = problem
        self.classes = classes
        self.classCounts = None
        self.splitFunctionValue = split_f_v
        self.splitFunction = lambda x: True
        self.feature_threshold = None

        self.left_ends = []
        self.right_ends = []
        self.widths = []

        if self.parent is not None:
            self.problem = self.parent.problem

            self.left_ends = self.parent.left_ends[:]
            self.right_ends = self.parent.right_ends[:]
            self.widths = self.parent.widths[:]
            index, t = self.parent.feature_threshold
            if self.splitFunctionValue:
                self.right_ends[index] = t
            else:
                self.left_ends[index] = t
            self.widths[index] = self.right_ends[index] - self.left_ends[index]

    def split(self, data):
        ys = data[:, 1]
        if self.homogeneous(ys) or any([w < 0.001 for w in self.widths]):
            return self.majorityVote(data)

        indices = self.generate_indices()
        index_threshs = self.generate_splits(indices)
        if index_threshs == []:
            return self.majorityVote(data)
        best_it = max(index_threshs, key=lambda it_pair: self.gain(data, it_pair))

        if self.gain(data, best_it) < 0.001:
            return self.majorityVote(data)

        self.splitFunction = lambda x: x[best_it[0]] < best_it[1]
        self.feature_threshold = best_it

        # add child nodes and process recursively
        for dataSubset in splitData(data, best_it):
            aChild = ContinuousTree(parent=self, split_f_v=self.splitFunction(dataSubset[0][0]))
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

    def gain(self, data, it_pair):
        if self.problem == 'r':
            var_gain = np.std(data[:, 1])
            for data_subset in splitData(data, it_pair):
                if data_subset.size == 0: return 0
                var_gain -= (data_subset.shape[0]) / (data.shape[0]) * np.std(data_subset[:, 1]) #ys
            return var_gain
        else:
            entropyGain = entropy(dataToDistribution(data))
            for dataSubset in splitData(data, it_pair):
                entropyGain -= len(dataSubset)/len(data)*entropy(dataToDistribution(dataSubset))
            return entropyGain


    def test(self, data):
        result = TestResult(self.classes)
        for x,y in data:
            result.update(self.predict(x), y)
        result.calculate_accuracy()
        return result


    def majorityVote(self, data):
        ''' Label node with the majority of the class labels in the given data set. '''
        if self.problem == 'r':
            self.label = np.mean(data[:, 1])
            return self
        else:
            labels = [label for (pt, label) in data]
            choice = max(set(labels), key=labels.count)
            self.label = choice
            self.classCounts = dict([(label, labels.count(label)) for label in set(labels)])
            return self


    def homogeneous(self, data):
        ''' Return True if the data have the same label, and False otherwise. '''
        if self.problem == 'r':
            return np.std(data) <= 0.1
        else:
            return len(data) <= 5

    def read_data(file):
        with open(file, 'r') as inputFile:
            lines = inputFile.readlines()
        data = [line.strip().split(",") for line in lines]
        data = [np.array([np.array([float(n) for n in x[:-1]]), x[-1]]) for x in data]
        random.shuffle(data)
        data = np.array(data)
        length = int(0.8 * len(data))
        train = data[:length, :]
        test = data[length:, :]
        return train, test


def splitData(data, it_pair):
    index, threshold = it_pair
    for res in [True, False]:
        data_subset = np.array([row for row in data
                                if (row[0][index] < threshold) == res])
        yield data_subset



def dataToDistribution(data):
    ''' Turn a dataset which has n possible classification labels into a
    probability distribution with n entries. '''
    allLabels = [label for (point, label) in data]
    numEntries = len(allLabels)
    possibleLabels = set(allLabels)

    dist = []
    for aLabel in possibleLabels:
        dist.append(allLabels.count(aLabel) / numEntries)
    return dist

def entropy(dist):
    ''' Compute the Shannon entropy of the given probability distribution. '''
    return -sum([p * math.log(p, 2) for p in dist])