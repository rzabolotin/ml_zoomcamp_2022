import requests

url = 'http://localhost:8001'
url = 'http://ml-churn-env.eba-bra9dhbe.us-east-1.elasticbeanstalk.com'

client = {"reports": 0, "share": 0.245, "expenditure": 3.438, "owner": "yes"}

result = requests.post(f"{url}/predict", json=client).json()
print(result)
