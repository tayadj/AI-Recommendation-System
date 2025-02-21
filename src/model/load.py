import src.core.config
import src.util

import torch
import sys
import os

logger = src.util.log.ModelLogger



def load(version):

    """
        Loads the model from the storage.

        Args:
            - version (str): The version of the model to load.

        Returns:
            - dict: The model, environment and config loaded from the specified version.
    """

    if version.lower() not in src.core.config.Config['available_model']:

        logger.error(f"model.load({version}): Wrong model - \"{version}\", expected - {src.core.config.Config['available_model']}.")
        raise src.util.exception.DataException(f"data.load({version}): Wrong model - \"{version}\", expected - {src.core.config.Config['available_model']}.")
    
    path = os.path.dirname(__file__) + '\\storage\\' + version.lower()
    
    data = torch.load(path, weights_only = False)

    logger.info(f"model.load({version}): model \"{version}\" loading, return - {data}.")
    
    return data