import pandas
import re



class DataValidationPipeline:

    def __init__(self, config = {}):

        self.version = config.get('version')

    def validate(self, text):

        match self.version:

            case 'alpha':

                stops = {'own', 'will', 'same', "you'll", 'few', "should've", 'nor', 'until', 'here', "haven't", 'below', 'then', 'very', 'it', 'of', 'll', 'shouldn', 'they', 'during', 'at', 'themselves', "won't", 'out', 'such', 'his', 'shan', 'its', 've', 'be', 'while', 'won', 'up', 'he', 'the', 't', 'didn', "isn't", 'myself', 'them', "that'll", 'after', 'aren', 'an', 'what', 'needn', 'those', 'was', 'further', 'ain', 'for', "you'd", 'there', "you're", 'a', 'hers', 'on', 'once', 'am', "wasn't", 'herself', 'having', 'any', 'theirs', 'with', 'because', 'mightn', 'if', 'before', 'or', 'ma', 'yourself', 'have', 'against', 'did', 'yours', 'are', 'whom', 'you', "don't", 'isn', 'how', 'her', 'why', "weren't", 'she', 'doing', 'mustn', "shan't", 'haven', 'to', 'each', "didn't", 'than', 'as', 'who', 'from', "she's", 'should', 'our', 'is', "mustn't", 'weren', 'were', 's', 'where', 'we', 'hasn', "it's", 'your', 'about', 'hadn', 'that', 'under', "you've", 'my', "wouldn't", 'and', 'don', 'too', 'me', 'so', 're', 'does', 'both', "hadn't", 'him', 'only', 'other', "hasn't", 'down', 'has', 'over', 'when', 'by', 'off', "shouldn't", 'all', 'itself', 'their', 'above', 'y', 'ourselves', 'some', 'himself', 'doesn', 'no', 'through', 'more', 'being', 'between', 'm', 'which', 'ours', 'had', 'can', "couldn't", 'just', 'wasn', 'this', "needn't", 'been', 'wouldn', "mightn't", 'yourselves', 'd', 'but', 'in', 'most', 'into', 'i', "doesn't", 'couldn', 'o', 'do', "aren't", 'again', 'these', 'now'}

                text = re.sub(r'<[^>]+>', '', text, flags=re.MULTILINE)
                text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
                text = re.sub(r'\[\d+\]|&#91;\d+&#93;', '', text)
                text = re.sub(r'&#\d+;', '', text)
                text = re.sub(r'([{}])'.format(re.escape("!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'")), r' \1 ', text)
                text = re.sub(r'\d+', '', text)
                text = re.sub(r'\b\w\b', '', text)
                text = re.sub(r'[^\w\s]', '', text)
                text = re.sub(r'\s+', ' ', text)
                text = text.strip()
                text = text.lower()     
                text = ' '.join([word for word in text.split() if word not in stops])

        return text

    def process(self, data):

        match self.version: 

            case 'alpha':

                data['text']['message'] = data['text']['message'].map(self.validate)

        return data
