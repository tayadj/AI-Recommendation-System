import pandas
import numpy
import torch



class ModelEmbeddingPipeline:

	def __init__(self, data_subject, data_object, data_action):

		self.data_subject = data_subject
		self.data_object = data_object
		self.data_action = data_action

	def featuring(self):

		current_date = pandas.to_datetime('today')
		self.data_subject['age'] = self.data_subject['birth'].apply(lambda value: current_date.year - value.year - ((current_date.month, current_date.day) < (value.month, value.day)))

	def process(self):

		self.featuring()

		return self.data_subject, self.data_object, self.data_action

		