import src.data
import pandas
import sys
import os



class DataIngestionPipeline:

	def __init__(self, version):

		self.version = version

	def process(self, data, config = {}):

		mode = config.get('mode', 'new')

		match mode:

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