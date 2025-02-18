import src.data
import pandas
import sys
import os



class DataIngestionPipeline:

	def __init__(self):

		pass

	def ingestion(self, data, config = {}):

		mode = config.get('mode', 'new')
		version = config.get('version')

		match mode:

			case 'new':

				pass

			case 'append':

				pass

			case _:

				pass

