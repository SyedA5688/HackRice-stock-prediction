import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
import matplotlib.pyplot as plt
from util import csv_to_dataset, history_days
import numpy as np
np.random.seed(4)
tf.random.set_seed(4)


# Load preprocessed dataset of stock prices
ohlcv_histories, next_day_open_values, unscaled_y, y_normaliser = csv_to_dataset('MSFT_daily.csv')

test_split = 0.9 # Training sample percentage as a decimal
n = int(ohlcv_histories.shape[0] * test_split)

# Split dataset into train and test sets
ohlcv_train = ohlcv_histories[:n]
open_prices_train = next_day_open_values[:n]

ohlcv_test = ohlcv_histories[n:]
open_prices_test = next_day_open_values[n:]

unscaled_open_prices_test = unscaled_y[n:]


# Build model
lstm_input = Input(shape=(history_days, 5), name='lstm_input')
x = LSTM(50, name='lstm_0')(lstm_input)
x = Dropout(0.2, name='lstm_dropout_0')(x)
x = Dense(64, name='dense_0')(x)
x = Activation('sigmoid', name='sigmoid_0')(x)
x = Dense(1, name='dense_1')(x)
output = Activation('linear', name='linear_output')(x)
model = Model(inputs=lstm_input, outputs=output)

adam = optimizers.Adam(lr=0.0005) # Optimizer that will alter gradients
model.compile(optimizer=adam, loss='mse')
model.fit(x=ohlcv_train, y=open_prices_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)


# Evaluate model
open_prices_test_predicted = model.predict(ohlcv_test)
# Scale values back up to regular values
open_prices_test_predicted = y_normaliser.inverse_transform(open_prices_test_predicted)
# Predictions from entire dataset
open_prices_predicted = model.predict(ohlcv_histories)
open_prices_predicted = y_normaliser.inverse_transform(open_prices_predicted)

assert unscaled_open_prices_test.shape == open_prices_test_predicted.shape
real_mse = np.mean(np.square(unscaled_open_prices_test - open_prices_test_predicted))
scaled_mse = real_mse / (np.max(unscaled_open_prices_test) - np.min(unscaled_open_prices_test)) * 100
print(scaled_mse)


# Plot real and predicted stock prices to visualize model's performance
plt.gcf().set_size_inches(22, 15, forward=True)
start = 0
end = -1
real = plt.plot(unscaled_open_prices_test[start:end], label='real')
pred = plt.plot(open_prices_test_predicted[start:end], label='predicted')

plt.legend(['Real', 'Predicted'])
plt.show()

model.save(f'basic_model.h5')


