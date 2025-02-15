import sklearn
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
        

	def __init__(self, data_subject, data_object, data_action, config = {}):

		self.data_subject = data_subject
		self.data_object = data_object
		self.data_action = data_action

		self.batch_size = config.get('batch_size', 1)

	def featuring(self):

		current_date = pandas.to_datetime('today')
		self.data_subject['age'] = self.data_subject['birth'].apply(lambda value: current_date.year - value.year - ((current_date.month, current_date.day) < (value.month, value.day)))

	def encode(self):

		encoder_gender = sklearn.preprocessing.LabelEncoder()
		self.data_subject['gender'] = encoder_gender.fit_transform(self.data_subject['gender'])

		encoder_location = sklearn.preprocessing.LabelEncoder()
		self.data_subject['location'] = encoder_gender.fit_transform(self.data_subject['location'])

		encoder_category = sklearn.preprocessing.LabelEncoder()
		self.data_object['category'] = encoder_gender.fit_transform(self.data_object['category'])

	def merge(self):

		self.data_subject = self.data_subject.add_prefix('subject_')
		self.data_object = self.data_object.add_prefix('object_')

		self.data = pandas.merge(self.data_action, self.data_subject, left_on = 'subject_id', right_on = 'subject_id', suffixes = ('_action', '_subject'))
		self.data = pandas.merge(self.data, self.data_object, left_on = 'object_id', right_on = 'object_id', suffixes = ('_subject', '_object'))

	def describe(self):

		self.config = {}

		self.config['dimension_subject'] = self.data['subject_id'].nunique()
		self.config['dimension_object'] = self.data['object_id'].nunique()
		self.config['dimension_gender'] = self.data['subject_gender'].nunique()
		self.config['dimension_location'] = self.data['subject_location'].nunique()
		self.config['dimension_category'] = self.data['object_category'].nunique()

	def process(self):

		self.featuring()
		self.encode()
		self.merge()
		self.describe()

		self.dataset = self.Dataset(self.data)
		self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size = self.batch_size, shuffle = True)

		return self.config, self.dataloader

		