import requests

r = requests.get("http://127.0.0.1:1234/models")
print(r.json())
