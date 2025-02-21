import pytest
import pandas
import sys

sys.path.append(sys.path[0]+"\\..")

import src

@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("query, prediction", [
    ("I love it!", 1),
    ("Terrible service.", -1),
    ("Meeting is quite ordinary, nothing special, just mediocre..", 0),
    ("Awful, waste of money", -1),
    ("The item is a dream come true, highly recommended!", 1)
])
def test_ModelAlpha(query, prediction):

    data = {'message': [query]}
    data = pandas.DataFrame(data)

    model_inference_pipeline = src.core.pipeline.ModelInferencePipeline("alpha")
    response = model_inference_pipeline.process(data)
    response = response.tolist()[0][0]

    assert abs(response - prediction) < 0.5