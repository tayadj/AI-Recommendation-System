from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas
import sys

sys.path.append(sys.path[0]+"\\..")

import src as RecSys



application = FastAPI(
    title = "AI Recommendation System"
)



class InferenceRequest(BaseModel):

    version: str
    config: Dict[str, str]
    data: Dict[str, List[Any]]

@application.post("/inference")
def inference(request: InferenceRequest):

    """
    Endpoint: 

        POST /inference

    Request:

        Headers:
            Content-Type: application/json

        Body Parameters:
            version (string): Version of the model to use for inference.
            config (object): Dictionary of configuration parameters.
            data (object): Dictionary of features, where each key is a feature name and the value is a list of corresponding feature values.

        Request body:
        {
	        "version": "sequential",
	        "config": {},
	        "data": {
		        "subject_id": [ 4, 1 ],
		        "subject_gender": [ "m", "m" ],
		        "subject_birth": [ "1988-06-30", "2014-09-11" ],
		        "subject_location": [ "Tokyo", "Berlin" ],
		        "object_id": [ 3, 1 ],
		        "object_category": [ "Sport", "Sport" ],
		        "timestamp": [ "2026-02-17", "2026-02-17" ]
	        }
        }

    Response:
    
        Status Code: 200 OK
        Response Body:
        {
            "inference": [[-0.49940162897109985],[0.9917365908622742]]
        }

        Status Code: 500 Internal Server error
        Response Body:
        {
            "detail": "Error message"
        }

    """

    try:

        version = request.version
        config = request.config
        data = request.data
        data = pandas.DataFrame(data)

        model_inference_pipeline = RecSys.core.pipeline.ModelInferencePipeline(version)
        response = model_inference_pipeline.process(data)
        response = response.tolist()

        return { "inference": response }

    except Exception as exception:

        raise HTTPException(status_code = 500, detail = str(exception))




class IngestionRequest(BaseModel):

    version: str
    config: Dict[str, str]
    data: Dict[str, Dict[str, List[Any]]]

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

    except Exception as exception:

        raise HTTPException(status_code = 500, detail = str(exception))

    

@application.get("/health")
def health():

    status = {
        "status": "OK"
    }

    return status



if __name__ == "__main__":

    import uvicorn
    uvicorn.run(application, host = "0.0.0.0", port = 8000)
