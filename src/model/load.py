import torch
import sys
import os



def load(version):

	path = os.path.dirname(__file__) + '\\storage\\' + version

	data = torch.load(path, weights_only = False)

	return data