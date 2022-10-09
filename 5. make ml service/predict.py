import pickle

new_client = {"reports": 0, "share": 0.001694, "expenditure": 0.12, "owner": "yes"}

model_path = 'model/model1.bin'
dv_path = 'model/dv.bin'

with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

with open(dv_path, "rb") as dv_file:
    dv = pickle.load(dv_file)



X = dv.transform(new_client)

predict = model.predict_proba(X)[0,1]
print(round(predict, 3))

