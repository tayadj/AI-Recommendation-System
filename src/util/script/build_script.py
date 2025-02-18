import src.core
import src.model
import src.data
import string
import pandas
import torch
import sys
import os
import re

def BuildScript(model_version, data_version = "base"):

    data = src.data.load(data_version)

    df_s = data['data'][0]
    df_o = data['data'][1]
    df_a = data['data'][2]

    dvp = src.core.pipeline.DataValidationPipeline(df_s, df_o, df_a, { 'exclude': ['id', 'subject_id', 'object_id', 'birth', 'rate', 'timestamp'], 'time': ['birth', 'timestamp']})
    df_clean_s, df_clean_o, df_clean_a = dvp.process()

    mep = src.core.pipeline.ModelEmbeddingPipeline(df_clean_s,df_clean_o,df_clean_a, src.core.Config)
    dc, dl = mep.process()
    encoder_gender, encoder_category, encoder_location = mep.encoder_gender, mep.encoder_category, mep.encoder_location
    engine = src.core.Engine()
    model = engine.produce(model_version)
    mtp = src.core.pipeline.ModelTrainingPipeline(model, dl)
    mtp.train()


    src.model.save(mtp.model, {'encoder_gender': encoder_gender, 'encoder_category': encoder_category, 'encoder_location': encoder_location}, 
    {'version': model_version, 'dvp_config': { 'exclude': ['id', 'subject_id', 'object_id', 'subject_birth', 'rate', 'timestamp'], 'time': ['subject_birth', 'timestamp'] }})