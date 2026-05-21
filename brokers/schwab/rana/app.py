import os
import json 
import pandas as pd 
import base64
import requests
from loguru import logger
from refresh_tokens import refresh_tokens 

with open('brokers/schwab/rana/token.json', 'r') as file:
    data = json.load(file)

base_url = "https://api.schwabapi.com/trader/v1"
headers = {"Authorization": f"Bearer {data['access_token']}"}
logger.info(f"access token:\n{data['access_token']}")
response = requests.get(
    base_url + f"/accounts/accountNumbers", headers=headers
)
response_frame = pd.json_normalize(response.json())
logger.info(f"response:\n{response_frame}")
accounthashed = response_frame['hashValue'][0] 
logger.info(f"{accounthashed}")
response1 = requests.get(
    base_url + f"/accounts/{accounthashed}", headers=headers
)
response1_frame = pd.json_normalize(response1.json())
response1_frame.to_csv('./response1_frame.csv') 
logger.info(f"response:\n{response1.json()}")
