import bentoml

loaded_model = bentoml.sklearn.load_model("mlzoomcamp_homework:qtzdz3slg6mwwdu5")

print((loaded_model.predict([[6.4,3.5,4.5,1.2]])))