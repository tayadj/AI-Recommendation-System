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

	def __init__(self, config = {}):

		self.version = config.get('version')

		match self.version:

			case 'alpha':

				self.batch_size = config.get('batch_size', 32)
				self.message_length = config.get('message_length', 128)

	def process(self, data):

		match self.version:

			case 'alpha':
			
				self.data = data['text']
				vocabulary = [token for message in self.data['message'] for token in message.split()]
				self.encoder = sklearn.preprocessing.LabelEncoder()
				self.encoder.fit(vocabulary)

				self.dataset = self.Dataset(self.data, {'version': 'alpha', 'encoder': self.encoder, 'message_length': self.message_length})
				self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size = self.batch_size, shuffle = True)

		return self.dataloader

		