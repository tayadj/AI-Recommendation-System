import torch
import sys
import os



def save(model, environment, config):

	"""
		Saves the Model to the storage.

		Args:
			- model (torch.nn.Module): The model to be saved.
			- environment (dict): The model environment for inference.
			- config (dict): Configuration dictionary containing additional information.
	"""

	path = os.path.dirname(__file__) + '\\storage\\' + config.get('version')

	data = {
		'model': model.state_dict(),
		'environment': environment,
		'config': config
	}
	
	torch.save(data, path)