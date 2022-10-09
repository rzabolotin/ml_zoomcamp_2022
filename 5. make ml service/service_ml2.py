from flask import Flask, request, jsonify
import pickle


app = Flask('PING')

model_path = 'model2.bin'
dv_path = 'dv.bin'

with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

with open(dv_path, "rb") as dv_file:
    dv = pickle.load(dv_file)

@app.route('/predict', methods=['POST'])
def predict():
    new_data = request.get_json()
    X = dv.transform(new_data)
    churn_probability = model.predict_proba(X)[0,1]
    return(jsonify({'churn_probability': churn_probability}))

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9595)
