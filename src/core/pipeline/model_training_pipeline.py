import src.core.config
import src.util

import torch

logger = src.util.log.Logger('core_logger', 'core.log')



class ModelTrainingPipeline:

	def __init__(self, model, dataloader, config = {}):

		self.model = model
		self.dataloader = dataloader
		self.version = config.get('version')

		if self.version.lower() not in src.core.config.Config['available_model']:

			logger.error(f"core.pipeline.ModelTrainingPipeline.__init__({model}, {dataloader}, {config}): Wrong model - \"{self.version}\", expected - {src.core.config.Config['available_model']}.")
			raise src.util.exception.CoreException(f"core.pipeline.ModelTrainingPipeline.__init__({model}, {dataloader}, {config}): Wrong model - \"{self.version}\", expected - {src.core.config.Config['available_model']}.")

		self.epochs = config.get('epochs', 5)
		self.learning_rate = config.get('learning_rate', 0.001)
		self.device = config.get('device', 'cpu')

		self.criterion = torch.nn.MSELoss()
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)

		logger.info(f"core.pipeline.ModelTrainingPipeline.__init__({model}, {dataloader}, {config}): model training pipeline initialisation for model \"{self.version}\".")

	def train_step(self):

		match self.version.lower():

			case 'alpha':

				self.model.train()

				loss_rate = 0.0
		
				for batch in self.dataloader:
		
					inputs = batch['message'].to(self.device)
					targets = batch['tone'].to(self.device).float()

					self.optimizer.zero_grad()
					outputs = self.model(inputs).squeeze()
					targets = targets.view_as(outputs)
					loss = self.criterion(outputs, targets)
					loss.backward()
					self.optimizer.step()

					loss_rate += loss.item() * inputs.size(0)

				loss_rate /= len(self.dataloader.dataset)

		logger.debug(f"core.pipeline.ModelTrainingPipeline.train_step(): model training pipeline train step for model \"{self.version}\", return - {loss_rate}.")

		return loss_rate				

	def train(self):
		
		self.model.to(self.device)

		for epoch in range(self.epochs):

			loss_rate = self.train_step()
			print(f'Epoch {epoch+1}/{self.epochs}, Loss: {loss_rate:.4f}')

		logger.info(f"core.pipeline.ModelTrainingPipeline.train(): model training pipeline train for model \"{self.version}\" finish.")
