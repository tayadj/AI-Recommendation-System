import src.core.config
import src.util

import torch
import sys
import os

logger = src.util.log.DataLogger



def save(dataframe, config):

	"""
		Saves the data to the storage.

		Args:
			- dataframe (pandas.DataFrame): The data to be saved.
			- config (dict): Configuration dictionary containing additional information.
	"""

	version = config.get('version')

	if version.lower() not in src.core.config.Config['available_data']:

		logger.error(f"data.save({dataframe}, {config}): Wrong data - \"{version}\", expected - {src.core.config.Config['available_data']}.")
		raise src.util.exception.DataException(f"data.save({dataframe}, {config}): Wrong data - \"{version}\", expected - {src.core.config.Config['available_data']}.")


	path = os.path.dirname(__file__) + '\\storage\\' + version.lower()

	data = {
		'data': dataframe,
		'config': config
	}

	torch.save(data, path)

	logger.info(f"data.save({dataframe}, {config}): data \"{version}\" saving.")