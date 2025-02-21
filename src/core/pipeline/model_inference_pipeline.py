import src

import pandas
import numpy
import torch
import re



class ModelInferencePipeline:

	def __init__(self, version):

		self.data = src.model.load(version)

		self.engine = src.core.Engine()
		self.model = self.engine.produce(version)
		self.model.load_state_dict(self.data['model'])
		self.model.eval()

	def process(self, sample):

		match self.data['config'].get('version'):

			case 'alpha':

				data_validation_pipeline = src.core.pipeline.DataValidationPipeline({'version': 'alpha'})
				sample['message'] = sample['message'].map(data_validation_pipeline.validate)

				encoder = self.data['environment'].get('encoder')
				message_length = self.data['config'].get('message_length')

				input = []

				for record in sample['message']:

					message_tensor = []

					for value in record.split():

						try:

							message_tensor.append(encoder.transform([value])[0])

						except ValueError:

							message_tensor.append(0)

					message_tensor = numpy.array(message_tensor)

					if len(message_tensor) < message_length:

						message_tensor = torch.cat([torch.tensor(message_tensor, dtype = torch.long), torch.zeros(message_length - len(message_tensor), dtype = torch.long)])

					else:

						message_tensor = torch.tensor(message_tensor[:message_length], dtype = torch.long)

					input.append(message_tensor)

				return self.model.predict(torch.stack(input, dim = 0))
