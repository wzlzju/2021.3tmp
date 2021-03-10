import requests
import json

user_info = {'name': 'letian', 'password': '123'}
headers = {
    "Content-Type": "application/json;charset=utf8"
}
r = requests.post("http://10.186.54.191:5000/recommend", data=json.dumps(user_info), headers=headers)

# print(r.text)