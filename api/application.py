import sys

sys.path.append(sys.path[0]+'\\..')



import src as RecSys

help(RecSys)

#
# Note: Scripts are needed to be moved.
#



#
# Train Example Script
#

df_s, df_o, df_a = RecSys.data.load(sys.path[0] + '\\..\\src\\data\\storage')

dvp = RecSys.core.pipeline.DataValidationPipeline(df_s, df_o, df_a, { 'exclude': ['id', 'subject_id', 'object_id', 'birth', 'rate', 'timestamp'], 'time': ['birth', 'timestamp']})
df_clean_s, df_clean_o, df_clean_a = dvp.process()

mep = RecSys.core.pipeline.ModelEmbeddingPipeline(df_clean_s,df_clean_o,df_clean_a, RecSys.core.config.Config)
dc, dl = mep.process()
dc['batch_size'] = RecSys.core.config.Config['batch_size']
encoder_gender, encoder_location, encoder_category = mep.encoder_gender, mep.encoder_location, mep.encoder_category

engine = RecSys.core.engine.Engine()
model = engine.produce("base")
mtp = RecSys.core.pipeline.ModelTrainingPipeline(model, dl)
mtp.train()

#
# Inference Example Script
#
#
#
# Note: move to model inference pipeline
#

'''

import pandas
import torch

mtp.load_model(sys.path[0]+'\\..\\src\\model\\storage\\model_100')
model = mtp.model
model.eval() 

# imitation of data from extrenal API
data = pandas.DataFrame({
    'subject_id': [1],
    'subject_gender': ['m'],
    'subject_birth': ['2014-09-11'],
    'subject_location': ['Berlin'],
    'object_id': [1],
    'object_category': ['sport'],
    'timestamp': ['2026-02-17']
})

dvp = RecSys.core.pipeline.DataValidationPipeline(exclude = ['subject_id', 'object_id', 'subject_birth', 'timestamp'], time = ['subject_birth', 'timestamp'])
data_processed = dvp.process(data)

current_date = pandas.to_datetime('today')
data_processed['subject_age'] = data_processed['subject_birth'].apply(lambda value: current_date.year - value.year - ((current_date.month, current_date.day) < (value.month, value.day)))
data_processed['subject_gender'] = encoder_gender.transform(data_processed['subject_gender'])
data_processed['subject_location'] = encoder_location.transform(data_processed['subject_location'])
data_processed['object_category'] = encoder_category.transform(data_processed['object_category'])

input = {
    'subject_id': torch.tensor(data_processed['subject_id'], dtype=torch.long),
    'subject_gender': torch.tensor(data_processed['subject_gender'], dtype=torch.long),
    'subject_age': torch.tensor(data_processed['subject_age'], dtype=torch.float),
    'subject_location': torch.tensor(data_processed['subject_location'], dtype=torch.long),
    'object_id': torch.tensor(data_processed['object_id'], dtype=torch.long),
    'object_category': torch.tensor(data_processed['object_category'], dtype=torch.long),
}

with torch.no_grad():

    prediction = model.predict(input)
    print(f'Prediction: {prediction.item()}')

'''