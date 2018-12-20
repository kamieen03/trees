from RandomForest import RandomForest
from RegressionTree import RegressionTree
from TestResult import TestResult
import random
import numpy as np
from matplotlib import pyplot as plt

def read_data():
    with open('concrete.txt', 'r') as inputFile:
        lines = inputFile.readlines()
    data = [line.strip().split() for line in lines]
    data = [(np.array([float(n) for n in x[:-1]]), float(x[-1])) for x in data]
    random.shuffle(data)
    data = np.array(data)
    length = int(0.8 * len(data))
    train = data[:length, :]
    test = data[length:, :]
    return train, test


if __name__ == '__main__':
    RMSEs, MAEs = [], []
    SIZE = 20
    xs = [i for i in range(1, SIZE + 1)]
    for i in range(10):
        train, test = read_data()
        f = RandomForest(0, train, test, "regression")
        tempRMSEs, tempMAEs = [], []
        for j in range(SIZE):
            t = RegressionTree(part_of_forest = True)
            train = f.sample(f.train_data)
            #train = f.train_data
            t.learn(train)
            f.trees.append(t)

            ts = TestResult(problem="regression")
            for x, y in test:
                #print("prediction", x, t.predict(x), y)
                ts.update(f.classify(x), y)
            ts.calculate_accuracy()
            tempRMSEs.append(ts.RMSE)
            tempMAEs.append(ts.MAE)
            print("{}-th iteration of {}-th test finished".format(j, i), ts.RMSE, ts.MAE)
        RMSEs.append(tempRMSEs)
        MAEs.append(tempMAEs)
    ysRMSE = np.mean(RMSEs, axis = 0)
    ysMAE = np.mean(MAEs, axis = 0)
    fig, ax  = plt.subplots()
    ax.plot(xs, ysRMSE, label = "RMSE")
    ax.plot(xs, ysMAE, label = "MAE")
    ax.set_ylabel("Number of trees in forest")
    ax.legend()
    plt.show()