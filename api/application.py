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



class BuildRequest(BaseModel):

    version: str

@application.post("/build")
def build(request: BuildRequest):

    try:

        version = request.version
        RecSys.util.script.BuildScript(version)

        return {"status": "OK"}

    except Exception:

        raise HTTPException(status_code = 500, detail = str(Exception))

    

@application.get("/health")
def health():

    status = {
        "status": "OK"
    }

    return status



if __name__ == "__main__":

    import uvicorn
    uvicorn.run(application, host = "0.0.0.0", port = 8000)
