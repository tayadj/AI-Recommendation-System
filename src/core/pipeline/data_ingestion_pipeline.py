import src.data
import src.util

import pandas
import sys
import os

logger = src.util.log.Logger('core_logger', 'core.log')



class DataIngestionPipeline:

	def __init__(self, version):

		self.version = version

		logger.info(f"core.pipeline.DataIngestionPipeline.__init__({version}): data ingestion pipeline initialisation.")

	def process(self, data, config = {}):

		mode = config.get('mode', 'new')

		match mode.lower():

			case 'new':

				config['version'] = self.version
				config.pop('mode')
				src.data.save(data, config)

			case 'append':

				loaded = src.data.load(self.version)
				storage = loaded['data']
				config = loaded['config']

				concatenated = {}

				for key, value in storage.items():

					concatenated[key] = pandas.concat([value, data[key]], ignore_index = True)

				src.data.save(concatenated, config)

		logger.info(f"core.pipeline.DataIngestionPipeline.process({data}, {config}): data ingestion pipeline process in mode \"{mode}\".")