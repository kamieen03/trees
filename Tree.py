import math
import numpy as np
import random
from itertools import chain
from collections import defaultdict

class Tree:
    graphViz="digraph G{\n"
    def __init__(self, parent=None):
        self.parent = parent
        self.children = []
        self.label = None
        self.classCounts = None
        self.splitFeatureValue = None
        self.splitFeature = None
        
    def to_string(self):
        if self.children == []:
            return "\""+str(id(self)) + "|"+str(self.splitFeatureValue) + "_"+ str(self.label)+"_"+str(self.classCounts)+"\""
        else:
            return "\""+str(id(self)) + "|"+str(self.splitFeatureValue) + "_"+ str(self.splitFeature)+"\""

    def add_to_graphViz(self):
        for child in self.children:
            #print(Tree.graphViz)
            Tree.graphViz += self.to_string() + " -> " + child.to_string() + "\n"
            child.add_to_graphViz()

    def to_graphViz(self):
        self.add_to_graphViz()
        Tree.graphViz+="}"
        return Tree.graphViz

    def split(self, data, remainingFeatures):
        ''' Build a decision tree from the given data, appending the children
        to the given root node (which may be the root of a subtree). '''
        if homogeneous(data):
            self.label = data[0][1]
            self.classCounts = {self.label: len(data)}
            return self

        if len(remainingFeatures) == 0:
            return majorityVote(data, self)

        # find the index of the best feature to split on
        a = list(remainingFeatures)
        random.shuffle(a)
        split_features = a[:self.split_features_num]
        bestFeature = max(split_features, key=lambda index: gain(data, index))

        if gain(data, bestFeature) == 0:
            return majorityVote(data, self)

        self.splitFeature = bestFeature

        # add child nodes and process recursively
        for dataSubset in splitData(data, bestFeature):
            aChild = Tree(parent=self)
            aChild.splitFeatureValue = dataSubset[0][0][bestFeature]
            self.children.append(aChild)

            aChild.split(dataSubset, remainingFeatures - set([bestFeature]))
        return self

    def learn(self, data):
        self.split(data, set(range(len(data[0][0]))))

    def classify(self, point):
        if self.children == []:
            return self.label
        else:
            matchingChildren = [child for child in self.children
                if child.splitFeatureValue == point[self.splitFeature]]
            if len(matchingChildren) == 0:
                return self.children[0].label
            return matchingChildren[0].classify(point)

    def test_Tree(self, data):
        correct = 0
        for x,y in data:
            if self.classify(x) == y:
                correct+=1
        return correct/len(data)*100

    def prune(self):
        if self.children == []:
            if len(self.parent.children) == 0:
                return
            elif len(self.parent.children) == 1:
                self.parent.classCounts = self.classCounts
                self.parent.feature = None
                self.parent.label = self.label
                self.parent.children = []
                self.parent.prune()
            elif self.parent.children[0].children == [] and self.parent.children[1].children == []:
                if sum([sum(list(child.classCounts.values())) for child in self.parent.children]) < 10:
                    node = self.parent
                    d = defaultdict(list)
                    d1 = node.children[0].classCounts
                    d2 = node.children[1].classCounts
                    for k, v in chain(d1.items(), d2.items()):
                        d[k].append(v)
                    d = dict(d)
                    for k in d:
                        d[k] = sum(d[k])
                    node.classCounts = d
                    node.feature = None
                    node.label = max(node.classCounts, key = lambda k: node.classCounts[k])
                    node.children = []
                    node.prune()
        else:
            for child in self.children: child.prune()




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

def splitData(data, featureIndex):
    ''' Iterate over the subsets of data corresponding to each value
    of the feature at the index featureIndex. '''
    attrValues = [point[featureIndex] for (point, label) in data]
    for aValue in set(attrValues):
        # compute the piece of the split corresponding to the chosen value
        dataSubset = [(point, label) for (point, label) in data
                    if point[featureIndex] == aValue]

        yield dataSubset

def gain(data, featureIndex):
    ''' Compute the expected gain from splitting the data along all possible
    values of feature. '''
    entropyGain = entropy(dataToDistribution(data))
    for dataSubset in splitData(data, featureIndex):
        entropyGain -= len(dataSubset)/len(data)*entropy(dataToDistribution(dataSubset))
    return entropyGain

def homogeneous(data):
    ''' Return True if the data have the same label, and False otherwise. '''
    return len(set([label for (point, label) in data])) <= 1

def majorityVote(data, node):
    ''' Label node with the majority of the class labels in the given data set. '''
    labels = set([label for (pt, label) in data])
    choice = max(labels, key=labels.count)
    node.label = choice
    node.classCounts = dict([(label, labels.count(label)) for label in labels])
    return node



def read_data():
    with open('obrazy2/obrazy2.txt', 'r') as inputFile:
        lines = inputFile.readlines()
    data = [line.split() for line in lines]
    data = [(np.array([int(n) for n in x[0]]), x[1]) for x in data]
    random.shuffle(data)
    length = int(0.8 * len(data))
    train = data[:length]
    test = data[length:]
    return train, test

