from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas
import sys

sys.path.append(sys.path[0]+"\\..")

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
        data = request.model_dump()
        data.pop("version")
        data = pandas.DataFrame(data)

        model_inference_pipeline = RecSys.core.pipeline.ModelInferencePipeline(version)
        response = model_inference_pipeline.process(data)
        response = response.tolist()

        return {"inference": response}

    except Exception:

        raise HTTPException(status_code = 500, detail = str(Exception))



class IngestionRequest(BaseModel):

    pass

@application.post("/ingestion")
def ingestion(request: IngestionRequest):

    pass



@application.get("/health")
def health():

    status = {
        "status": "OK"
    }

    return status



if __name__ == "__main__":

    import uvicorn
    uvicorn.run(application, host = "0.0.0.0", port = 8000)



#
# Note: Scripts are needed to be moved.
#



def build(version):

    #
    # Train Example Script
    #

    data= RecSys.data.load('base')
    df_s = data['data'][0]
    df_o = data['data'][1]
    df_a = data['data'][2]

    dvp = RecSys.core.pipeline.DataValidationPipeline(df_s, df_o, df_a, { 'exclude': ['id', 'subject_id', 'object_id', 'birth', 'rate', 'timestamp'], 'time': ['birth', 'timestamp']})
    df_clean_s, df_clean_o, df_clean_a = dvp.process()

    mep = RecSys.core.pipeline.ModelEmbeddingPipeline(df_clean_s,df_clean_o,df_clean_a, RecSys.core.config.Config)
    dc, dl = mep.process()
    encoder_gender, encoder_category, encoder_location = mep.encoder_gender, mep.encoder_category, mep.encoder_location

    engine = RecSys.core.engine.Engine()
    model = engine.produce(version)
    mtp = RecSys.core.pipeline.ModelTrainingPipeline(model, dl)
    mtp.train()


    RecSys.model.save(mtp.model, {'encoder_gender': encoder_gender, 'encoder_category': encoder_category, 'encoder_location': encoder_location}, 
    {'version': version, 'dvp_config': { 'exclude': ['id', 'subject_id', 'object_id', 'subject_birth', 'rate', 'timestamp'], 'time': ['subject_birth', 'timestamp'] }})





