from alpha_vantage.timeseries import TimeSeries
import json

def save_dataset(symbol):
  credentialsObj = json.load(open('keys.json', 'r'))
  api_key = credentialsObj['AV_api_key']

  ts = TimeSeries(key=api_key, output_format='pandas')
  data, meta_data = ts.get_daily(symbol, outputsize='full')

  data.to_csv(f'./{symbol}_daily.csv')

save_dataset('AAPL')