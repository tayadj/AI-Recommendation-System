import src.core
import src.model
import src.util

import pandas
import numpy
import torch
import re

logger = src.util.log.Logger('core_logger', 'core.log')



class ModelInferencePipeline:

	def __init__(self, version):

		if version.lower() not in src.core.config.Config['available_model']:

			logger.error(f"core.pipeline.ModelInferencePipeline.__init__({version}): Wrong model - \"{version}\", expected - {src.core.config.Config['available_model']}.")
			raise src.util.exception.CoreException(f"core.pipeline.ModelInferencePipeline.__init__({version}): Wrong model - \"{version}\", expected - {src.core.config.Config['available_model']}.")

		self.data = src.model.load(version)

		self.engine = src.core.Engine()
		self.model = self.engine.produce(version)
		self.model.load_state_dict(self.data['model'])
		self.model.eval()

		logger.info(f"core.pipeline.ModelInferencePipeline.__init__({version}): model inference pipeline initialisation for model \"{version}\".")

	def process(self, sample):

		match self.data['config'].get('version').lower():

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

				prediction = self.model.predict(torch.stack(input, dim = 0))

		logger.info(f"core.pipeline.ModelInferencePipeline.process({sample}): model inference pipeline process for model \"{self.data['config'].get('version').lower()}\", return - {prediction}.")

		return prediction
