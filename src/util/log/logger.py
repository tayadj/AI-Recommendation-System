import logging

def Logger(name, storage, level = logging.INFO):

	formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

	handler = logging.FileHandler(storage)
	handler.setFormatter(formatter)

	logger = logging.getLogger(name)
	logger.setLevel(level)
	logger.addHandler(handler)

	return logger