import torch
import sys
import os



def save(dataframe, config):

    """
		Saves the data to the storage.

		Args:
		dataframe (pandas.DataFrame): The data to be saved.
		config (dict): Configuration dictionary containing additional information.
    """

	path = os.path.dirname(__file__) + '\\storage\\' + config.get('version')

	data = {
		'data': dataframe,
		'config': config
	}

	torch.save(data, path)