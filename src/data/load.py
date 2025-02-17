import pandas
import torch
import sys
import os



def load(version):

    path = os.path.dirname(__file__) + '\\storage\\' + version

    data = torch.load(path, weights_only = False)

    data_subject = data['data'][0]
    data_object = data['data'][1]
    data_action = data['data'][2]

    return data_subject, data_object, data_action
