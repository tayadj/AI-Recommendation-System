from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas
import sys

sys.path.append(sys.path[0]+"\\..")

import src



application = FastAPI(
    title = "AI System"
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

        model_inference_pipeline = src.core.pipeline.ModelInferencePipeline(version)
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

    """
    Endpoint: 

        POST /ingestion

    Request:

        Headers:
            Content-Type: application/json

        Body Parameters:
            version (string): Version of the data to be ingested.
            config (object): Dictionary of configuration parameters.
            data (object): Dictionary of dataframes, where each key is a dataframe name and the value is a dictionary of corresponding data.

        Request body:
        {
	        "version": "base",
	        "config": {
                "mode": "append"
            },
	        "data": {
                "subject": {
                    "id": [100, 101, 102],
                    "gender": [ "m", "m", "f" ],
                    "birth": [ "2001-01-01", "1982-12-10", "2003-04-21" ],
                    "location": [ "Tokyo", "Paris", "Moscow" ]
                },
                "object": {
                    "id": [100],
                    "name": ["Sunflower seeds"],
                    "description": ["This seeds are ideal for your garden"],
                    "category": ["Biology"]
                },
                "action": {
                    subject_id: [100],
                    object_id: [3],
                    rate: [0.35],
                    timestamp: [ "2025-02-19" ]
                }
	        }
        }

    Response:
    
        Status Code: 200 OK
        Response Body:
        {
            "status": "OK"
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

        dataframe = {}

        for key, value in data.items():

            dataframe[key] = pandas.DataFrame(value)

        data_ingestion_pipeline = src.core.pipeline.DataIngestionPipeline(version)
        data_ingestion_pipeline.process(dataframe, config)

        return { "status": "OK" }

    except Exception as exception:

        raise HTTPException(status_code = 500, detail = str(exception))



class BuildRequest(BaseModel):

    model: str
    data: str

@application.post("/build")
def build(request: BuildRequest):

    """
    Endpoint: 

        POST /build

    Request:

        Headers:
            Content-Type: application/json

        Body Parameters:
            model (string): Version of the model to be built.
            data (string): Version of the data to be used.

        Request Body:
        {
            "model": "sequential",
            "data": "base"
        }

    Response:
    
        Status Code: 200 OK
        Response Body:
        {
            "status": "OK"
        }

        Status Code: 500 Internal Server error
        Response Body:
        {
            "detail": "Error message"
        }
    """

    try:

        model = request.model
        data = request.data
        src.util.script.BuildScript(model, data)

        return { "status": "OK" }

    except Exception as exception:

        raise HTTPException(status_code = 500, detail = str(exception))

    

@application.get("/health")
def health():

    """
    Endpoint: 

        GET /health 

    Response:
    
        Status Code: 200 OK
        Response Body:
        {
            "status": "OK"
        }
    """

    return { "status": "OK" }



if __name__ == "__main__":

    import uvicorn
    uvicorn.run(application, host = "0.0.0.0", port = 8000)
