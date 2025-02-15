import pandas
import numpy
import torch



class ModelEmbeddingPipeline:

	class Dataset(torch.utils.data.Dataset):

		def __init__(self, data):

			self.data = data

		def __len__(self):

			return len(self.data)

		def __getitem__(self, index):

			record = self.data.iloc[index]

			return {
				'subject_id': record['subject_id'],
				'subject_gender': record['subject_gender'],
				'subject_age': record['subject_age'],
				'subject_location': record['subject_location'],
				'object_id': record['object_id'],
				'object_category': record['object_category'],
				'rate': record['rate'],
			}
        

	def __init__(self, data_subject, data_object, data_action):

		self.data_subject = data_subject
		self.data_object = data_object
		self.data_action = data_action

	def featuring(self):

		current_date = pandas.to_datetime('today')
		self.data_subject['age'] = self.data_subject['birth'].apply(lambda value: current_date.year - value.year - ((current_date.month, current_date.day) < (value.month, value.day)))

	def merge(self):

		self.data_subject = self.data_subject.add_prefix('subject_')
		self.data_object = self.data_object.add_prefix('object_')

		self.data = pandas.merge(self.data_action, self.data_subject, left_on = 'subject_id', right_on = 'subject_id', suffixes = ('_action', '_subject'))
		self.data = pandas.merge(self.data, self.data_object, left_on = 'object_id', right_on = 'object_id', suffixes = ('_subject', '_object'))

	def process(self):

		self.featuring()
		self.merge()

		self.dataset = self.Dataset(self.data)
		self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size = 1, shuffle = True)

		return self.dataloader

		