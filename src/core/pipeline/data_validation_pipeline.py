import pandas
import string
import re



class DataValidationPipeline:

    def __init__(self, data_subject, data_object, data_action, config = {}):

        self.data_subject = data_subject
        self.data_object = data_object
        self.data_action = data_action

        self.exlude = config.get('exclude', [])
        self.time = config.get('time', [])

    def clean(self, text):

        text = re.sub(r'([{}])'.format(re.escape(string.punctuation)), r' \1 ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\[\d+\]|&#91;\d+&#93;', '', text)
        text = re.sub(r'&#\d+;', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = text.strip()
        text = text.lower()

        return text

    def impute(self, text):

        return text

    def process(self):

        for data in [self.data_subject, self.data_object, self.data_action]:

            for attribute in data.columns:

                if attribute not in self.exlude:

                    data[attribute] = data[attribute].map(str)
                    data[attribute] = data[attribute].map(self.clean)
                    data[attribute] = data[attribute].map(self.impute)

                if attribute in self.time:

                    data[attribute] = pandas.to_datetime(data[attribute], format='%Y-%m-%d')

        return self.data_subject, self.data_object, self.data_action
