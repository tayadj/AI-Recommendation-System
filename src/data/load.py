import src.core.config
import src.util

import pandas
import torch
import sys
import os

logger = src.util.log.DataLogger



def load(version):

    """
        Loads the data from the storage.

        Args:
            - version (str): The version of the data to load.

        Returns:
            - dict: The data and config loaded from the specified version.
    """

    if version.lower() not in src.core.config.Config['available_data']:

            logger.error(f"data.load({version}): Wrong data - \"{version}\", expected - {src.core.config.Config['available_data']}.")
            raise src.util.exception.DataException(f"data.load({version}): Wrong data - \"{version}\", expected - {src.core.config.Config['available_data']}.")

    path = os.path.dirname(__file__) + '\\storage\\' + version.lower()

    data = torch.load(path, weights_only = False)

    logger.info(f"data.load({version}): data \"{version}\" loading, return - {data}.")

    return data
