import src.core
import src.model
import src.data
import string
import pandas
import torch
import sys
import os
import re

def BuildScript(model_version, data_version):

    data = src.data.load(data_version)

    match data_version:

        case 'alpha':

            data_validation_pipeline = src.core.pipeline.DataValidationPipeline({'version': 'alpha'})
            data_clean = data_validation_pipeline.process(data['data'])

            model_embedding_pipeline = src.core.pipeline.ModelEmbeddingPipeline({'version': 'alpha'})
            data_loader = model_embedding_pipeline.process(data_clean)
            encoder = model_embedding_pipeline.encoder

            engine = src.core.Engine()
            model = engine.produce('alpha')

            for record in data_loader:
                
                print(record)

        case 'base':

            dataframe_subject = data['data'][0]
            dataframe_object = data['data'][1]
            dataframe_action = data['data'][2]

            data_validation_pipeline = src.core.pipeline.DataValidationPipeline({'version': data_version})
            data_clean = data_validation_pipeline.process([data['data'][0], data['data'][1], data['data'][2]], { 'exclude': ['id', 'subject_id', 'object_id', 'birth', 'rate', 'timestamp'], 'time': ['birth', 'timestamp'] })
            
            mep = src.core.pipeline.ModelEmbeddingPipeline({'version': data_version})
            dl = mep.process(data_clean)
            encoder_gender, encoder_category, encoder_location = mep.encoder_gender, mep.encoder_category, mep.encoder_location
            engine = src.core.Engine()
            model = engine.produce(model_version)
            mtp = src.core.pipeline.ModelTrainingPipeline(model, dl, {'version': data_version})
            mtp.train()

            src.model.save(mtp.model, {'encoder_gender': encoder_gender, 'encoder_category': encoder_category, 'encoder_location': encoder_location}, 
            {'version': model_version, 'dvp_config': { 'exclude': ['id', 'subject_id', 'object_id', 'subject_birth', 'rate', 'timestamp'], 'time': ['subject_birth', 'timestamp'] }})

            # Needed to add model and data versions to src.model.save