import torch
import sys
import os



def load(version):

    """
        Loads the model from the storage.

        Args:
            - version (str): The version of the model to load.

        Returns:
            - dict: The model, environment and config loaded from the specified version.
    """

	path = os.path.dirname(__file__) + '\\storage\\' + version

	data = torch.load(path, weights_only = False)

	return data