import torch

def save(model, environment, config):

	data = {
		'model': model,
		'environment': environment,
		'config': config
	}

	torch.save('./storage/' + config.get('version'))