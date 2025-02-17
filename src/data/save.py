import torch
import sys
import os



def save(data_subject, data_object, data_action, config):

	path = os.path.dirname(__file__) + '\\storage\\' + config.get('version')

	data = {
		'data': [data_subject, data_object, data_action],
		'config': config
	}

	torch.save(data, path)