import torch

class Engine():

	class Model(torch.nn.Module):

		def __init__(self, config = {}):
	
			super(Engine.Model, self).__init__()

			self.dimension_subject = config.get('dimension_subject', 1024)
			self.dimension_object = config.get('dimension_object', 1024)
			self.dimension_gender = config.get('dimension_gender', 2)
			self.dimension_location = config.get('dimension_location', 128)
			self.dimension_category = config.get('dimension_category', 128)
			self.batch_size = config.get('batch_size', 4)

			self.embedding_subject = torch.nn.Embedding(self.dimension_subject, 128)
			self.embedding_object = torch.nn.Embedding(self.dimension_object, 128)
			self.embedding_gender = torch.nn.Embedding(self.dimension_gender, 8)
			self.embedding_location = torch.nn.Embedding(self.dimension_location, 32)
			self.embedding_category = torch.nn.Embedding(self.dimension_category, 32)

			self.network = torch.nn.Sequential(
				torch.nn.Linear(128 + 128 + 8 + 32 + 32, 512),
				torch.nn.ReLU(),
				torch.nn.Linear(512, 256),
				torch.nn.ReLU(),
				torch.nn.Linear(256, 1),
				torch.nn.Tanh()
			)

		def forward(self, x):

			subject_embedded = self.embedding_subject(x['subject_id'])
			object_embedded = self.embedding_object(x['object_id'])
			gender_embedded = self.embedding_gender(x['subject_gender'])
			location_embedded = self.embedding_location(x['subject_location'])
			category_embedded = self.embedding_category(x['object_category'])

			embedding = torch.cat((subject_embedded, object_embedded, gender_embedded, location_embedded, category_embedded), dim=1)
			embedding = embedding.view(embedding.size(0), -1)

			return self.network(embedding)

		def predict(self, x):

			return self.forward(x)



	def __init__(self):
	
		pass

	def produce(self, mode, config = {}):

		match mode:

			case "base":

				return self.Model(config)

			case _:

				return self.Model(config)
