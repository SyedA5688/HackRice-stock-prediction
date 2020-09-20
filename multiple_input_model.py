import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
import matplotlib.pyplot as plt
from util import csv_to_dataset, history_days, multiple_csv_to_dataset
import numpy as np
np.random.seed(4)
tf.random.set_seed(4)


# Load preprocessed dataset of stock prices
#ohlcv_histories, moving_averages, next_day_open_values, unscaled_open_prices, y_normaliser = csv_to_dataset('MSFT_daily.csv')

# Split into test and training sets
#train_split = 0.8
#n = int(ohlcv_histories.shape[0] * train_split)
#ohlcv_train = ohlcv_histories[:n]
#mov_avg_train = moving_averages[:n]
#open_prices_train = next_day_open_values[:n]
#ohlcv_test = ohlcv_histories[n:]
#mov_avg_test = moving_averages[n:]
#open_prices_test = next_day_open_values[n:]
#unscaled_open_prices_test = unscaled_open_prices[n:]


# Multiple csv dataset function returns training and testing splits already, commented out above
# Training set is now Microsoft, Netflix, and Facebook stock prices. Google stock prices is the test set
ohlcv_train, mov_avg_train, open_prices_train, ohlcv_test, mov_avg_test, open_prices_test, unscaled_open_prices_test, y_normaliser = multiple_csv_to_dataset('GOOGL_daily.csv')
#**mov_avg_train is size of original moving_averages, ohlcv_train is size of old ohlvc_histories


# Build Model v2 - more complex layers, 2 inputs
# Two sets of input into model - previous stock prices over time and the techincal indicator (moving average)
lstm_input = Input(shape=(history_days, 5), name='lstm_input')
dense_input = Input(shape=(mov_avg_train.shape[1],), name='tech_input')
 
# First branch of model has layers for first input, stock prices from data
x = LSTM(50, name='lstm_0')(lstm_input)
x = Dropout(0.2, name='lstm_dropout_0')(x)
lstm_branch = Model(inputs=lstm_input, outputs=x)
 
# Second branch - Moving Average technical indicator input
y = Dense(20, name='tech_dense_0')(dense_input)
y = Activation("relu", name='tech_relu_0')(y)
y = Dropout(0.2, name='tech_dropout_0')(y)
moving_averages_branch = Model(inputs=dense_input, outputs=y)
 
# Combine two branches
combined_branches = concatenate([lstm_branch.output, moving_averages_branch.output], name='concatenate')
z = Dense(64, activation="sigmoid", name='dense_pooling')(combined_branches)
z = Dense(1, activation="linear", name='dense_out')(z)
 
# Model takes inputs from both branches, outputs a single value
model = Model(inputs=[lstm_branch.input, moving_averages_branch.input], outputs=z)
adam = optimizers.Adam(lr=0.0005)
model.compile(optimizer=adam, loss='mse')
# Train model
model.fit(x=[ohlcv_train, mov_avg_train], y=open_prices_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)


# Evaluate model
# Open prices prediction
open_prices_test_predicted = model.predict([ohlcv_test, mov_avg_test])
open_prices_test_predicted = y_normaliser.inverse_transform(open_prices_test_predicted)

# Entire dataset prediction
open_prices_predicted = model.predict([ohlcv_train, mov_avg_train])
open_prices_predicted = y_normaliser.inverse_transform(open_prices_predicted)

assert unscaled_open_prices_test.shape == open_prices_test_predicted.shape
real_mse = np.mean(np.square(unscaled_open_prices_test - open_prices_test_predicted))
scaled_mse = real_mse / (np.max(unscaled_open_prices_test) - np.min(unscaled_open_prices_test)) * 100
print(scaled_mse)


# Plot Predictions and real values
plt.gcf().set_size_inches(22, 15, forward=True)
start = 0
end = -1
real = plt.plot(unscaled_open_prices_test[start:end], label='real')
pred = plt.plot(open_prices_test_predicted[start:end], label='predicted')

plt.legend(['Real', 'Predicted'])

plt.show()


model.save(f'multiple_input_multiple_datasets_model.h5')

