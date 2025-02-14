import torch

class Engine(torch.nn.Module):

	def __init__(self, config = {}):
	
		super(Engine, self).__init__()

		self.dimension_input = config.get('dimension_input', 32)
		self.dimension_hidden = config.get('dimension_hidden', 1024)
		self.dimension_output = config.get('dimension_output', 32)

		self.network = torch.nn.Sequential(
			torch.nn.Linear(self.dimension_input, self.dimension_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.dimension_hidden, self.dimension_output)
		)

	def forward(self, x):

		return self.network(x)

	def predict(self, x):

		return self.forward(x)
