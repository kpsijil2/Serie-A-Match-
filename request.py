import requests

url = 'https://localhost:5000/predict_api'
s = requests.post(url, json={'HomeTeam': 22, 'AwayTeam': 11, 'FTHG': 2, 'FTAG': 1, 'HS': 13, 'AS': 14, 'HST': 3, 'AST': 2, 'HC': 7,
                             'AC': 4, 'HR': 0, 'AR': 0})

print(s.json())
