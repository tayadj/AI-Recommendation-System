import torch
import sys
import os



def save(dataframe, config):

	path = os.path.dirname(__file__) + '\\storage\\' + config.get('version')

	data = {
		'data': dataframe,
		'config': config
	}

	torch.save(data, path)