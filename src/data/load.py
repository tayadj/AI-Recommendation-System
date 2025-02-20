import pandas
import torch
import sys
import os



def load(version):

    """
        Loads the data from the storage.

        Args:
        version (str): The version of the data to load.

        Returns:
        torch.Tensor: The data loaded from the specified version.
    """

    path = os.path.dirname(__file__) + '\\storage\\' + version

    data = torch.load(path, weights_only = False)

    return data
