import pandas
import string
import re



class DataValidationPipeline:

    def __init__(self, config = {}):

        self.version = config.get('version')

    def validate(self, text):

        match self.version:

            case 'alpha':

                text = re.sub(r'([{}])'.format(re.escape(string.punctuation)), r' \1 ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                text = re.sub(r'<[^>]+>', '', text)
                text = re.sub(r'\[\d+\]|&#91;\d+&#93;', '', text)
                text = re.sub(r'&#\d+;', '', text)
                text = re.sub(r'[^\w\s]', '', text)
                text = re.sub(r'\s+', ' ', text)
                text = text.strip()
                text = text.lower()     

        return text

    def process(self, data, config = {}):

        match self.version: 

            case 'alpha':

                data['text']['message'] = data['text']['message'].map(self.validate)

        return data
