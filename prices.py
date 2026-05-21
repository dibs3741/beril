import yfinance as yf 
import requests
import json
import pandas as pd 
from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

#data = yf.download('HASI', '2025-12-10', '2025-12-25') 
#data.reset_index(inplace=True)
#print(data) 
#print(data[['Date', 'Close']]) 

'''
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=CRSAX&apikey=RQ8C0BZI344OD74D'
session = requests.Session() 
#session.headers.update({ 
#    'X-API-Key': 'lYzBFq87UrUmVW_SCgGK9EfHo3kFsmGiJcysbcc0cjE' 
#}) 
response1 = session.get(url, verify=False) 
if response1.status_code != 200: 
    raise 
res_json = json.loads(response1.text) 

d = res_json['Time Series (Daily)']
li = []
da = {}
for key, value in d.items():
    da['asofdate'] = key
    da.update(value)
    print(da) 
    li.append(da) 
    da = {}
    #da.clear()
print(li)
df = pd.DataFrame(li)
print(df.to_string())

#with open("sample.json", "w") as f:     
#    f.write(response1.text)
#df_simple1 = pd.json_normalize(res_json['Time Series (Daily)'])
#df_simple1.to_csv('./df_simple1.csv')  
'''

url = f'https://eodhd.com/api/eod/CRSAX?api_token=699e7b7beb1948.60098083&fmt=json'
data = requests.get(url).json()
#res_json = json.loads(data) 
df = pd.DataFrame(data)
print(df) 
