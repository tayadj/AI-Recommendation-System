import re
import string

class DataValidationPipeline:

    def __init__(self):

        pass

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

        data = data.map(str)
        data = data.map(self.clean)
        data = data.map(self.impute)

        return data
        
