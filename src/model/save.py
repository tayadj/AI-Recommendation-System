import torch
import sys
import os



def save(model, environment, config):

	path = os.path.dirname(os.path.abspath(__file__)) + '\\storage\\' + config.get('version')

	data = {
		'model': model.state_dict(),
		'environment': environment,
		'config': config
	}

	torch.save(data, path)