import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import load_model
import matplotlib.dates as mdates

csv = pd.read_csv("stock_train.csv")
#print(csv)

data = csv.loc[:, ("Open")].values
#print(data)

#scale input between 0, 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.reshape(-1, 1))
#print(scaled_data)
#print(scaled_data.shape[0])

x = []
y = []
#y lags x by 50 samples
for i in range(50, scaled_data.shape[0]):
    x.append(scaled_data[i-50:i, 0])
    y.append(scaled_data[i, 0])

x, y = np.array(x), np.array(y)

x = np.reshape(x, (x.shape[0], x.shape[1], 1))
#print(x.shape)
#(1208, 50, 1)
#print(x[0])
print(y.shape)
#print(y)

#train on GPU
pysical_devices = tf.config.experimental.list_physical_devices('GPU')
#print("Num GPUs Available: ", len(pysical_devices))
tf.config.experimental.set_memory_growth(pysical_devices[0], True)
"""
model = Sequential()
model.add(LSTM(128, input_shape=(50, 1), return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.1))

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.1))

model.add(LSTM(128))
model.add(Dropout(0.1))

model.add(Dense(1))
model.summary()

model.compile(
    loss='mean_squared_error',
    optimizer='adam',
)

model.fit(x,
          y,
          epochs=100,
          batch_size=32)

#model.save("stock.h5")
"""
#append test.csv to train.csv
csv2 = pd.read_csv("stock_test.csv")
#print(csv2)
csv3 = pd.concat((csv, csv2), axis=0)

#get last 50 from train.csv plus all data from test.csv
data = csv3.loc[:, ("Open")].iloc[x.shape[0]:, ].values
#print(data.shape)
#print(data)

scaled_data = scaler.fit_transform(data.reshape(-1, 1))

x_test = []

for i in range(50, scaled_data.shape[0]):
    x_test.append(scaled_data[i-50:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#print(x_test.shape)
#(20, 50, 1)

model = load_model('stock.h5')

predicted_stock_price = model.predict(x_test)
#print(predicted_stock_price.shape)
#print(predicted_stock_price)

predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
#rint(predicted_stock_price)

ax = plt.figure(figsize=(7, 5), dpi=100).add_subplot(111)

t = csv2.loc[:, "Date"].values
t = pd.to_datetime(t)

ax.plot(t, csv2.loc[:, ("Open")].values, label="Real stock price", color="red")
ax.plot(t, predicted_stock_price, label="Predicted Stock Price", color="blue")

ax.legend()
ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d %Y'))

ax.set_title("Stock Price Prediction")
plt.ylabel("USD")

ax.xaxis.grid(True, which="major")
ax.yaxis.grid(True, which="major")

plt.xticks(rotation=45)
plt.subplots_adjust(bottom=.2)
plt.show()

