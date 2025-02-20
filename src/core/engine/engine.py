import torch



class Engine():

	class ModelAlpha(torch.nn.Module):

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

		def forward(self, x):

			embedded = self.embedding(x)
			LSTM_output, (hidden, cell) = self.LSTM(embedded)
			hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1) if self.bidirectional else hidden[-1,:,:]
			output = self.dense(hidden)
			return self.Tanh(output)

		def predict(self, x):

			with torch.no_grad():

				self.eval()
				prediction = self.forward(x)
				self.train()

				return prediction

	def __init__(self):
	
		pass

	def produce(self, mode, config = {}):

		match mode.lower():

			case 'alpha':

				return self.ModelAlpha(config)