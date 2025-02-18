import torch
import sys
import os



def save(dataframes, config):

	path = os.path.dirname(__file__) + '\\storage\\' + config.get('version')

	data = {
		'data': dataframes,
		'config': config
	}

	torch.save(data, path)