import logging



def Logger(name, storage, level = logging.INFO):

	formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

	handler = logging.FileHandler('./log/' + storage)
	handler.setLevel(level)
	handler.setFormatter(formatter)

	logger = logging.getLogger(name)
	logger.setLevel(level)
	logger.addHandler(handler)

	return logger



CoreLogger = Logger('core_logger', 'core.log')
DataLogger = Logger('data_logger', 'data.log')
ModelLogger = Logger('model_logger', 'model.log')