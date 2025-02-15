import torch

class Engine(torch.nn.Module):

	def __init__(self, config = {}):
	
		super(Engine, self).__init__()

		self.dimension_subject = config.get('dimension_subject', 1024)
		self.dimension_object = config.get('dimension_object', 1024)
		self.dimension_gender = config.get('dimension_gender', 2)
		self.dimension_location = config.get('dimension_location', 128)
		self.dimension_category = config.get('dimension_category', 128)
		self.batch_size = config.get('batch_size', 1)


	def forward(self, x):

		return self.network(x)

	def predict(self, x):

		return self.forward(x)
