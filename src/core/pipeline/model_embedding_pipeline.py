import src.util
import src.core.config

import sklearn
import pandas
import numpy
import torch

logger = src.util.log.CoreLogger



class ModelEmbeddingPipeline:

	class Dataset(torch.utils.data.Dataset):

		def __init__(self, data, config):

			self.data = data
			self.version = config.get('version')
			
			match self.version.lower():

				case 'alpha':

					self.encoder = config.get('encoder')
					self.message_length = config.get('message_length')			

			logger.info(f"core.pipeline.ModelEmbeddingPipeline.Dataset.__init__(): data initialisation for model \"{self.version}\".")

		def __len__(self):

			logger.debug(f"core.pipeline.ModelEmbeddingPipeline.Dataset.__len__(): call dataset length for model \"{self.version}\", return - {len(self.data)}.")

			return len(self.data)

		def __getitem__(self, index):

			record = self.data.iloc[index]

			match self.version.lower():

				case 'alpha':

					message_tensor = self.encoder.transform(record['message'].split())

					if len(message_tensor) < self.message_length:

						message_tensor = torch.cat([torch.tensor(message_tensor, dtype = torch.long), torch.zeros(self.message_length - len(message_tensor), dtype = torch.long)])

					else:

						message_tensor = torch.tensor(message_tensor[:self.message_length], dtype = torch.long)

					item = {
						'message': message_tensor,
						'tone': torch.tensor(record['tone'], dtype = torch.float)
					}

			logger.debug(f"core.pipeline.ModelEmbeddingPipeline.Dataset.__getitem__({index}): call dataset item for model \"{self.version}\", return - {item}.")

			return item

	def __init__(self, config = {}):

		self.version = config.get('version')

		match self.version.lower():

			case 'alpha':

				self.batch_size = config.get('batch_size', 32)
				self.message_length = config.get('message_length', 128)

			case _:

				logger.error(f"core.pipeline.ModelEmbeddingPipeline.__init__({config}): Wrong model - \"{self.version}\", expected - {src.core.config.Config['available_model']}.")
				raise src.util.exception.CoreException(f"core.pipeline.ModelEmbeddingPipeline.__init__({config}): Wrong model - \"{self.version}\", expected - {src.core.config.Config['available_model']}.")	
				
		logger.info(f"core.pipeline.ModelEmbeddingPipeline.__init__({config}): model embedding pipeline initialisation for model \"{self.version}\".")

	def process(self, data):

		match self.version.lower():

			case 'alpha':
			
				self.data = data['text']
				vocabulary = [token for message in self.data['message'] for token in message.split()]
				self.encoder = sklearn.preprocessing.LabelEncoder()
				self.encoder.fit(vocabulary)

				self.dataset = self.Dataset(self.data, {'version': 'alpha', 'encoder': self.encoder, 'message_length': self.message_length})
				self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size = self.batch_size, shuffle = True)

		logger.info(f"core.pipeline.ModelEmbeddingPipeline.process({data}): model embedding pipeline process for model \"{self.version}\", return - {self.dataloader}.")

		return self.dataloader

		