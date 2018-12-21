from abc import ABC, abstractmethod

class AbstractTree(ABC):
	@abstractmethod
	def __init__(self, parent = None, part_of_forest = False, split_features_num = None):
		self.parent = parent
		self.PART_OF_FOREST = part_of_forest
		self.split_features_num = split_features_num
		self.label = None
		self.children = []
		if self.parent is not None:
			self.PART_OF_FOREST = self.parent.PART_OF_FOREST
			self.split_features_num = self.parent.split_features_num

	@abstractmethod
	def learn(self, training_set):
		pass

	@abstractmethod
	def predict(self, point):
		pass

	@abstractmethod
	def gain(self):
		pass

		
	@staticmethod
	@abstractmethod
	def read_data(file):
		pass