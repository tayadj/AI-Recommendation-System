import pandas
import string
import re



class DataValidationPipeline:

    def __init__(self, exclude = None):

        self.exlude = exclude if exclude else []

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

    def process(self, data):

        for attribute in data.columns:

            if attribute not in self.exlude:

                data[attribute] = data[attribute].map(str)
                data[attribute] = data[attribute].map(self.clean)
                data[attribute] = data[attribute].map(self.impute)

        return data
