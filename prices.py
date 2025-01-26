import yfinance as yf 

data = yf.download('HASI', '2024-12-10', '2024-12-25') 
data.reset_index(inplace=True)
print(data) 
print(data[['Date', 'Close']]) 
