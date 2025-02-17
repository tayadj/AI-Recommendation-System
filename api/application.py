from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas
import sys

sys.path.append(sys.path[0]+'\\..')

import src as RecSys

application = FastAPI(
    title = "AI Recommendation System"
)


class InferenceRequest(BaseModel):

    version: str
    subject_id: List[int]
    subject_gender: List[str]
    subject_birth: List[str]
    subject_location: List[str]
    object_id: List[int]
    object_category: List[str]
    timestamp: List[str]

@application.post("/inference")
def inference(request: InferenceRequest):

    try:

        version = request.version
        data = request.dict()
        data.pop('version')
        data = pandas.DataFrame(data)

        print(version, data)

        model_inference_pipeline = RecSys.core.pipeline.ModelInferencePipeline(version)

        return {"message": "Inference completed successfully"}

    except Exception:

        raise HTTPException(status_code = 500, detail = str(Exception))

if __name__ == "__main__":

    import uvicorn
    uvicorn.run(application, host = "0.0.0.0", port = 8000)


'''
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

mip = RecSys.core.pipeline.ModelInferencePipeline('base')

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


print(mip.process(sample))
'''




