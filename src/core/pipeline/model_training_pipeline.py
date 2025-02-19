import torch

class ModelTrainingPipeline:

	def __init__(self, model, dataloader, config = {}):

		self.model = model
		self.dataloader = dataloader
		self.version = config.get('version')

		self.epochs = config.get('epochs', 5)
		self.learning_rate = config.get('learning_rate', 0.001)
		self.device = config.get('device', 'cpu')

		self.criterion = torch.nn.MSELoss()
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)

	def train_step(self):

		match self.version:

			case 'base':

				self.model.train()

				loss_rate = 0.0
		
				for batch in self.dataloader:
		
					inputs = {key: value.to(self.device) for key, value in batch.items() if key != 'rate'}
					targets = batch['rate'].to(self.device).float()

					self.optimizer.zero_grad()
					outputs = self.model(inputs).squeeze()
					targets = targets.view_as(outputs)
					loss = self.criterion(outputs, targets)
					loss.backward()
					self.optimizer.step()

					loss_rate += loss.item() * inputs['subject_id'].size(0)

				loss_rate /= len(self.dataloader.dataset)

				return loss_rate

	def train(self):
		
		self.model.to(self.device)
		for epoch in range(self.epochs):

			loss_rate = self.train_step()
			print(f'Epoch {epoch+1}/{self.epochs}, Loss: {loss_rate:.4f}')

