import sklearn
import pandas
import numpy
import torch



class ModelEmbeddingPipeline:

	class Dataset(torch.utils.data.Dataset):

		def __init__(self, data, config):

			self.data = data
			self.version = config.get('version')
			
			match self.version:

				case 'alpha':

					self.encoder = config.get('encoder')
					self.message_length = config.get('message_length')

		def __len__(self):

			return len(self.data)

		def __getitem__(self, index):

			record = self.data.iloc[index]

			match self.version:

				case 'alpha':

					message_tensor = self.encoder.transform(record['message'].split())

					if len(message_tensor) < self.message_length:

						message_tensor = torch.cat([torch.tensor(message_tensor, dtype = torch.long), torch.zeros(self.message_length - len(message_tensor), dtype = torch.long)])

					else:

						message_tensor = torch.tensor(message_tensor[:self.message_length], dtype = torch.long)

					return {
						'message': message_tensor,
						'tone': torch.tensor(record['tone'], dtype = torch.float)
					}

				case 'base':

					return {
						'subject_id': torch.tensor(record['subject_id'], dtype=torch.long),
						'subject_gender': torch.tensor(record['subject_gender'], dtype=torch.long),
						'subject_age': torch.tensor(record['subject_age'], dtype=torch.float),
						'subject_location': torch.tensor(record['subject_location'], dtype=torch.long),
						'object_id': torch.tensor(record['object_id'], dtype=torch.long),
						'object_category': torch.tensor(record['object_category'], dtype=torch.long),
						'rate': torch.tensor(record['rate'], dtype=torch.float),
					}

            
        


	def __init__(self, config = {}):

		self.version = config.get('version')
		self.batch_size = config.get('batch_size', 4)

		match self.version:

			case 'alpha':

				self.message_length = config.get('message_length', 128)
				

	# featuring, encode, merge, describe functions - to delete
	# move its functionality to process

	def featuring(self):

		match self.version:

			case 'base':

				current_date = pandas.to_datetime('today')
				self.data_subject['age'] = self.data_subject['birth'].apply(lambda value: current_date.year - value.year - ((current_date.month, current_date.day) < (value.month, value.day)))

	def encode(self):

		match self.version:

			case 'base':

				self.encoder_gender = sklearn.preprocessing.LabelEncoder()
				self.encoder_location = sklearn.preprocessing.LabelEncoder()
				self.encoder_category = sklearn.preprocessing.LabelEncoder()

				self.data_subject['gender'] = self.encoder_gender.fit_transform(self.data_subject['gender'])
				self.data_subject['location'] = self.encoder_location.fit_transform(self.data_subject['location'])
				self.data_object['category'] = self.encoder_category.fit_transform(self.data_object['category'])

	def merge(self):

		match self.version:

			case 'base':

				self.data_subject = self.data_subject.add_prefix('subject_')
				self.data_object = self.data_object.add_prefix('object_')

				self.data = pandas.merge(self.data_action, self.data_subject, left_on = 'subject_id', right_on = 'subject_id', suffixes = ('_action', '_subject'))
				self.data = pandas.merge(self.data, self.data_object, left_on = 'object_id', right_on = 'object_id', suffixes = ('_subject', '_object'))

	def describe(self):

		match self.version:

			case 'base':

				self.config = {}

				self.config['dimension_subject'] = self.data['subject_id'].nunique()
				self.config['dimension_object'] = self.data['object_id'].nunique()
				self.config['dimension_gender'] = self.data['subject_gender'].nunique()
				self.config['dimension_location'] = self.data['subject_location'].nunique()
				self.config['dimension_category'] = self.data['object_category'].nunique()

	def process(self, data):

		match self.version:

			case 'alpha':
			
				self.data = data['text']
				vocabulary = [token for message in self.data['message'] for token in message.split()]
				self.encoder = sklearn.preprocessing.LabelEncoder()
				self.encoder.fit(vocabulary)

				self.dataset = self.Dataset(self.data, {'version': 'alpha', 'encoder': self.encoder, 'message_length': self.message_length})
				self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size = self.batch_size, shuffle = True)

			case 'base':

				self.data_subject = data[0]
				self.data_object = data[1]
				self.data_action = data[2]

				self.featuring()
				self.encode()
				self.merge()
				self.describe()

				self.dataset = self.Dataset(self.data, {'version': self.version})
				self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size = self.batch_size, shuffle = True)

		return self.dataloader

		