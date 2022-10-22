import numpy as np

import bentoml
from bentoml.io import NumpyNdarray

#mlzoomcamp_homework:qtzdz3slg6mwwdu5
#mlzoomcamp_homework:jsi67fslz6txydu5

my_runner = bentoml.sklearn.get("mlzoomcamp_homework:jsi67fslz6txydu5").to_runner()

svc = bentoml.Service("mlzoomcamp_classify", runners=[my_runner])


@svc.api(
    input=NumpyNdarray.from_sample(np.array([[4.9, 3.0, 1.4, 0.2]], dtype=np.double)),
    output=NumpyNdarray(),
)
def classify(input_series: np.ndarray) -> np.ndarray:
    return my_runner.predict.run(input_series)