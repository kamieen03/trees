from DiscreteTree import DiscreteTree
from ContinuousTree import ContinuousTree
from TestResult import TestResult
import pickle
import random
import numpy as np

class RandomForest():
    def __init__(self, n, train, tree_type, problem, classes = None):
        self.split_features_num = len(train[0][0])
        self.train_data = train
        self.iteration = 0
        self.problem = problem[0].lower()
        self.classes = classes
        if tree_type == "con":
            self.trees = [ContinuousTree(part_of_forest = True,
                        split_features_num = self.split_features_num, 
                        problem = self.problem) for _ in range(n)]
        else:
            self.trees = [DiscreteTree(part_of_forest = True,
                        split_features_num = self.split_features_num) for _ in range(n)]

    def learn(self):
        for t in self.trees:
            train = self.sample(self.train_data)
            t.learn(train)
            self.iteration += 1
            print("{}-th tree trained.".format(self.iteration))

    def sample(self, data):
        n = len(data)
        new_data = []
        for _ in range(n):
            new_data.append(data[random.randrange(n)])
        return np.array(new_data)

    def predict(self, x):
        if self.problem == 'r':
            return np.mean([t.predict(x) for t in self.trees])
        elif self.problem == 'c':
            vote_dict = {c:0 for c in self.classes}
            for t in self.trees:
                c = t.predict(x)
                vote_dict[c] += 1
            return max(vote_dict, key = lambda c: vote_dict[c])

    def test(self, data):
        result = TestResult(self.classes)
        for x,y in data:
            result.update(self.predict(x), y)
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