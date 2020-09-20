from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from keras.models import load_model

app = Flask(__name__)
api = Api(app)

# Load saved model
model = load_model('multiple_input_multiple_datasets_model.h5')

def get_model_prediction(dataset_filename_to_read):
  data = pd.read_csv(dataset_filename_to_read)
  data = data.iloc[::-1]
  data = data.drop('date', axis=1)
  data = data.values

  normalizing_scaler = preprocessing.MinMaxScaler()
  normalized_data = normalizing_scaler.fit_transform(data)

  # Get y_normalizer
  # Get unscaled stock open price from original file data
  next_day_open_values = np.array([data[:,0][i + 30].copy() for i in range(len(data) - 30)])
  next_day_open_values = np.expand_dims(next_day_open_values, -1)
  y_normaliser = preprocessing.MinMaxScaler()
  y_normaliser.fit(next_day_open_values)

  # Get most recent 30 stock market open prices to feed to model, as well as moving average of values
  ohlcv_one = normalized_data[-30:]
  mov_avg_one = np.mean(ohlcv_one[:,3])
  # Prepare values to be fed into model
  ohlcv_one = np.array([ohlcv_one])
  mov_avg_one = np.array([mov_avg_one])

  predicted_price_tomorrow = np.squeeze(y_normaliser.inverse_transform(model.predict([ohlcv_one, mov_avg_one])))
  return predicted_price_tomorrow


# Request argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')

class PredictStockPrice(Resource):
  def get(self):
    # use parser and find the user's query. Accepted query vals are GOOGL, MSFT, NFLX, and FB
    args = parser.parse_args()
    user_query = args['query']

    # Send parameter to function to generate model's prediction of stock price of company tomorrow
    fileName = user_query + "_daily.csv"
    next_day_opening_stock_price = get_model_prediction(fileName)
    next_day_opening_stock_price = int(next_day_opening_stock_price)
    
    # create JSON object
    output = {'stock_price_next_day': next_day_opening_stock_price}
    
    return output
  

# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictStockPrice, '/')


if __name__ == '__main__':
    app.run(debug=False)

