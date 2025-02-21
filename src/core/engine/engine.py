import src.core.config
import src.util

import torch
import logging

logger = src.util.log.CoreLogger



class Engine():

	"""
		Engine class that encapsulates various models.

		Available models:
			- Model Alpha, which is used for sentiment analysis.
	"""

	class ModelAlpha(torch.nn.Module):

		"""
			Model Alpha is a neural network model designed for sentiment analysis.
        
			Attributes:
				- dimension_embedding (int): Dimension of the embedding vectors.
				- dimension_hidden (int): Dimension of the hidden state in the LSTM.
				- dimension_output (int): Dimension of the output.
				- vocabulary_size (int): Size of the input vocabulary.
				- layers_number (int): Number of LSTM layers.
				- bidirectional (bool): bidirectional LSTM mode.
				- dropout (float): Dropout rate.
		"""

		def __init__(self, config = {}):

			super(Engine.ModelAlpha, self).__init__()

			self.dimension_embedding = config.get('dimension_embedding', 128)
			self.dimension_hidden = config.get('dimension_hidden', 128)
			self.dimension_output = config.get('dimension_output', 1)
			self.vocabulary_size = config.get('vocabulary_size', 10000)
			self.layers_number = config.get('layers_number', 3)
			self.bidirectional = config.get('bidirectional', True)
			self.dropout = config.get('dropout', 0.5)

			self.embedding = torch.nn.Embedding(self.vocabulary_size, self.dimension_embedding)
			self.LSTM = torch.nn.LSTM(
				self.dimension_embedding,
				self.dimension_hidden,
				num_layers = self.layers_number,
				bidirectional = self.bidirectional,
				dropout = self.dropout,
				batch_first = True
			)
			self.dense = torch.nn.Linear(self.dimension_hidden * 2 if self.bidirectional else self.dimension_hidden, self.dimension_output)
			self.Tanh = torch.nn.Tanh()

			logger.info(f"core.engine.Engine.ModelAlpha.__init__(): model alpha initialisation - {self}.")

		def forward(self, input):

			embedded = self.embedding(input)
			LSTM_output, (hidden, cell) = self.LSTM(embedded)
			hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1) if self.bidirectional else hidden[-1,:,:]
			raw_output = self.dense(hidden)
			output = self.Tanh(raw_output)

			logger.debug(f"core.engine.Engine.ModelAlpha.forward({input}): model alpha forward propagation, return - {output}.")

			return output

		def predict(self, input):

			with torch.no_grad():

				self.eval()
				prediction = self.forward(input)

			logger.debug(f"core.engine.Engine.ModelAlpha.predict({input}): model alpha prediction, return - {prediction}.")

			return prediction

	
	def __init__(self):

		logger.info(f"core.engine.Engine.__init__(): model engine initialisation.")

	def produce(self, mode, config = {}):

		"""
			Factory method to produce models.
        
			Args:
				- mode (str): The type of model to produce.
				- config (dict): Configuration dictionary for the model.
        
			Returns:
				- torch.nn.Module: An instance of the specified model.
		"""

		match mode.lower():

			case 'alpha':

				return self.ModelAlpha(config)

			case _:

				logger.error(f"core.engine.Engine.produce(): Wrong engine mode - \"{mode}\", expected - {src.core.config.Config['available_model']}.")
				raise src.util.exception.CoreException(f"core.engine.Engine.produce(): Wrong engine mode - \"{mode}\", expected - {src.core.config.Config['available_model']}..")			