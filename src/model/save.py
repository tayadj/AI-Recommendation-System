import src.core.config
import src.util

import torch
import sys
import os

logger = src.util.log.ModelLogger



def save(model, environment, config):

	"""
		Saves the Model to the storage.

		Args:
			- model (torch.nn.Module): The model to be saved.
			- environment (dict): The model environment for inference.
			- config (dict): Configuration dictionary containing additional information.
	"""
	
	version = config.get('version')

	if version.lower() not in src.core.config.Config['available_model']:

		logger.error(f"model.save({model}, {environment}, {config}): Wrong model - \"{version}\", expected - {src.core.config.Config['available_model']}.")
		raise src.util.exception.DataException(f"model.save({model}, {environment}, {config}): Wrong model - \"{version}\", expected - {src.core.config.Config['available_model']}.")

	path = os.path.dirname(__file__) + '\\storage\\' + version.lower()

	data = {
		'model': model.state_dict(),
		'environment': environment,
		'config': config
	}
	
	torch.save(data, path)

	logger.info(f"model.save({model}, {environment}, {config}): model \"{version}\" saving.")