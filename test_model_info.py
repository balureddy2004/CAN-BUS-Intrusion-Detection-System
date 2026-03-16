# test_model_info.py
import requests
# list models first to pick a name
rl = requests.get("http://localhost:5000/api/models")
print("models list:", rl.json())
# then request info for the top model (if any)
models = rl.json().get("models", [])
if models:
    name = models[0]["name"]
    ri = requests.get(f"http://localhost:5000/api/model_info/{name}")
    print("model_info:", ri.status_code, ri.json())
else:
    print("no models found")
