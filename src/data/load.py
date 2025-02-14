import pandas

def load(path = './data/raw'):
    
    """
    Load data from .tsv files.
    
    Parameters: 
        path - path to the data source directory.
    
    Return:
        data_subject - dataframe with uploaded data about subjects.
        data_object - dataframe with uploaded data about objects.
        data_action - dataframe with uploaded data about actions.
    """

    data_subject = pandas.read_csv(path + '/subject.tsv', sep = '\t')
    data_object = pandas.read_csv(path + '/object.tsv', sep = '\t')
    data_action = pandas.read_csv(path + '/action.tsv', sep = '\t')
    
    return data_subject, data_object, data_action
