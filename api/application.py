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
encoder_gender, encoder_category, encoder_location = mep.encoder_gender, mep.encoder_category, mep.encoder_location

engine = RecSys.core.engine.Engine()
model = engine.produce("base")
mtp = RecSys.core.pipeline.ModelTrainingPipeline(model, dl)
mtp.train()


RecSys.model.save(mtp.model, {'encoder_gender': encoder_gender, 'encoder_category': encoder_category, 'encoder_location': encoder_location}, {'version': 'base'})



#
# Inference Example Script
#
#
#
# Note: move to model inference pipeline
#

import pandas
import torch

# imitation of data from extrenal API
sample = pandas.DataFrame({
    'subject_id': [4, 1],
    'subject_gender': ['m', 'm'],
    'subject_birth': ['1988-06-30', '2014-09-11'],
    'subject_location': ['Tokyo', 'Berlin'],
    'object_id': [3, 1],
    'object_category': ['Sport', 'Sport'],
    'timestamp': ['2026-02-17', '2026-02-17']
})


data = RecSys.model.load('base')

engine = RecSys.core.engine.Engine()
model = engine.produce("base")
model.load_state_dict(data['model'])
model.eval()


# Config need to be saved on model.save
dvp_config = { 'exclude': ['id', 'subject_id', 'object_id', 'subject_birth', 'timestamp'], 'time': ['subject_birth', 'timestamp']}

for attribute in sample.columns:

    if attribute not in dvp_config.get('exclude'):

        sample[attribute] = sample[attribute].map(str)
        sample[attribute] = sample[attribute].map(dvp.clean)
        sample[attribute] = sample[attribute].map(dvp.impute)

    if attribute in dvp_config.get('time'):

        sample[attribute] = pandas.to_datetime(sample[attribute], format='%Y-%m-%d')

encoder_gender = data['environment'].get('encoder_gender')
encoder_location = data['environment'].get('encoder_location')
encoder_category = data['environment'].get('encoder_category')

current_date = pandas.to_datetime('today')
sample['subject_age'] = sample['subject_birth'].apply(lambda value: current_date.year - value.year - ((current_date.month, current_date.day) < (value.month, value.day)))
sample['subject_gender'] = encoder_gender.transform(sample['subject_gender'])
sample['subject_location'] = encoder_location.transform(sample['subject_location'])
sample['object_category'] = encoder_category.transform(sample['object_category'])

input = {
    'subject_id': torch.tensor(sample['subject_id'], dtype=torch.long),
    'subject_gender': torch.tensor(sample['subject_gender'], dtype=torch.long),
    'subject_age': torch.tensor(sample['subject_age'], dtype=torch.float),
    'subject_location': torch.tensor(sample['subject_location'], dtype=torch.long),
    'object_id': torch.tensor(sample['object_id'], dtype=torch.long),
    'object_category': torch.tensor(sample['object_category'], dtype=torch.long),
}

with torch.no_grad():

    prediction = model.predict(input)
    print(f'Prediction: {prediction}')
