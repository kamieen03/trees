from Tree import Tree
from RegressionTree import RegressionTree
from TestResult import TestResult
import pickle
import random
import numpy as np

class RandomForest():
    def __init__(self, n, train, test, problem):
        self.split_features_num = len(train[0][0])
        self.train_data = train
        self.test_data = test
        self.iteration = 0
        self.problem = problem.upper()
        #self.split_features_num = int(len(self.train_data[0][0])**0.5)
        if self.problem == "REGRESSION":
            self.trees = [RegressionTree(part_of_forest = True) for _ in range(n)]
        else:
            self.trees = [Tree() for _ in range(n)]

    def learn(self):
        for t in self.trees:
            #t.split_features_num = self.split_features_num
            train = self.sample(self.train_data)
            t.learn(train)
            #t.prune()
            self.iteration += 1
            print(self.iteration)

    def sample(self, data):
        n = len(data)
        new_data = []
        for _ in range(n):
            new_data.append(data[random.randrange(n)])
        return np.array(new_data)

    def classify(self, x):
        if self.problem == "REGRESSION":
            return np.mean([t.predict(x) for t in self.trees])
        elif self.problem == "CLASSIFY":
            vote_dict = {str(i):0 for i in range(6)}
            for t in self.trees:
                c = t.classify(x)
                vote_dict[c] += 1
            return max(vote_dict, key = lambda c: vote_dict[c])

    def test(self, data):
        result = TestResult()
        for x,y in data:
            result.update(self.classify(x), y)
        result.calculate_accuracy()
        return result

    def serialize(self):
        file = open("learned_forest.pickle", "wb")
        pickle.dump(self, file)
        file.close()

    @staticmethod
    def deserialize():
        file = open("learned_forest.pickle", "rb")
        t = pickle.load(file)
        file.close()
        return t