class TestResult():
    def __init__(self, problem = "classification"):
        self.class_seen = {str(i): 0 for i in range(6)}
        self.class_correct = {str(i): 0 for i in range(6)}
        self.class_accuracy = {}
        self.seen = 0
        self.correct = 0
        self.accuracy = 0
        self.problem = problem
        self.RMSE = 0
        self.MAE = 0

    def update(self, p_x, y):
        self.seen += 1
        if self.problem == "classification":
            self.class_seen[y] += 1
            if p_x == y:
                self.correct += 1
                self.class_correct[y] += 1
        else:
            self.RMSE += (p_x-y)**2
            self.MAE += abs(p_x-y)

    def calculate_accuracy(self):
        if self.problem == "classification":
            for c in self.class_correct:
                self.class_accuracy[c] = self.class_correct[c]/self.class_seen[c]*100
            self.accuracy = self.correct/self.seen*100
        else:
            self.RMSE = (self.RMSE/self.seen)**0.5
            self.MAE = self.MAE/self.seen

    def __str__(self):
        if self.problem == "classification":
            ret = "\n".join(["Accuracy on " + c+": " +
                str(self.class_accuracy[c]) for c in self.class_accuracy])
            ret +="\nTotal accuracy: " + str(self.accuracy)
            return ret
        else:
            return ("RMSE = " + str(self.RMSE) + "\n" + 
                "MAE = " + str(self.MAE))