# test_predict.py
import requests, json
payload = {"raw": {"can_id": 817, "dlc": 8, "data": [0,1,2,3,4,5,6,7], "ts_delta": 0.0}}
r = requests.post("http://localhost:5000/api/predict", json=payload, timeout=60)
print("status:", r.status_code)
print("headers:", r.headers.get("content-type"))
print("body:", r.text)
